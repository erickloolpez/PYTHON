import time
import numpy as np
import pyopencl as cl
from tqdm import tqdm
import json
import psutil

def get_system_info():
    """Obtiene información del sistema para monitoreo"""
    try:
        temp = psutil.sensors_temperatures()
        cpu_temp = temp['coretemp'][0].current if 'coretemp' in temp else 'N/A'
    except:
        cpu_temp = 'N/A'
    
    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()
    gpu_name = devices[0].name if devices else "No GPU detectada"
    
    return cpu_temp, ram_usage, gpu_name

def vectorizar_secuencias(secuencia1, secuencia2, max_len):
    """Prepara las secuencias para procesamiento en GPU"""
    sec1_padded = secuencia1.ljust(max_len)
    sec2_padded = secuencia2.ljust(max_len)
    arr1 = np.array(list(sec1_padded), dtype='S1')
    arr2 = np.array(list(sec2_padded), dtype='S1')
    return arr1, arr2

def comparar_secuencias_gpu(secuencia1, secuencia2, ctx, queue):
    """Compara dos secuencias completas usando GPU y devuelve solo diferencias"""
    max_len = max(len(secuencia1), len(secuencia2))
    arr1, arr2 = vectorizar_secuencias(secuencia1, secuencia2, max_len)
    
    mf = cl.mem_flags
    buf1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr1)
    buf2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr2)
    diferencias_buf = cl.Buffer(ctx, mf.WRITE_ONLY, max_len * 2 * np.dtype(np.int8).itemsize)
    
    # Kernel modificado para encontrar diferencias
    kernel_code = """
    __kernel void encontrar_diferencias(
        __global const char* sec1,
        __global const char* sec2,
        const int length,
        __global char* diferencias
    ) {
        int pos = get_global_id(0);
        
        if (pos < length) {
            char c1 = sec1[pos];
            char c2 = sec2[pos];
            
            // Solo marcar diferencias (0 si son iguales)
            if (c1 != c2 && c1 != ' ' && c2 != ' ') {
                diferencias[pos*2] = c1;    // Carácter del archivo 1
                diferencias[pos*2+1] = c2;  // Carácter del archivo 2
            } else {
                diferencias[pos*2] = 0;
                diferencias[pos*2+1] = 0;
            }
        }
    }
    """
    
    prg = cl.Program(ctx, kernel_code).build()
    prg.encontrar_diferencias(queue, (max_len,), None, buf1, buf2, np.int32(max_len), diferencias_buf)
    
    diferencias_gpu = np.empty(max_len * 2, dtype=np.int8)
    cl.enqueue_copy(queue, diferencias_gpu, diferencias_buf).wait()
    
    diferencias = []
    for pos in range(max_len):
        char1 = chr(diferencias_gpu[pos*2])
        char2 = chr(diferencias_gpu[pos*2+1])
        if char1 != '\x00' and char2 != '\x00':
            diferencias.append({
                'posicion': pos+1,  # +1 para posición basada en 1
                'arch1': char1,
                'arch2': char2
            })
    
    buf1.release()
    buf2.release()
    diferencias_buf.release()
    
    return diferencias

def procesar_archivos_gpu(ruta_archivo1, ruta_archivo2):
    """Procesa dos archivos FASTA comparando línea por línea usando GPU"""
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    def cargar_lineas(ruta):
        lineas = []
        with open(ruta, 'r') as f:
            for linea in f:
                linea = linea.strip()
                if linea and not linea.startswith('>'):
                    lineas.append(linea)
        return lineas
    
    lineas1 = cargar_lineas(ruta_archivo1)
    lineas2 = cargar_lineas(ruta_archivo2)
    total_lineas = min(len(lineas1), len(lineas2))
    
    json_file = open("diferencias_gpu.json", "w")
    txt_file = open("diferencias_gpu.txt", "w")
    txt_file.write("Diferencias encontradas:\n")
    txt_file.write(f"Archivo 1: {ruta_archivo1}\n")
    txt_file.write(f"Archivo 2: {ruta_archivo2}\n\n")
    
    total_diferencias = 0
    muestra_resultados = []
    
    with tqdm(total=total_lineas, desc="Procesando") as pbar:
        for i in range(total_lineas):
            resultado = {'linea': i+1}
            
            if lineas1[i] == lineas2[i]:
                resultado['resultado'] = 'identical'
                txt_file.write(f"Línea {i+1}: [CADENA IDÉNTICA]\n")
            else:
                diferencias = comparar_secuencias_gpu(lineas1[i], lineas2[i], ctx, queue)
                if diferencias:
                    resultado['diferencias'] = diferencias
                    total_diferencias += len(diferencias)
                    txt_file.write(f"Línea {i+1}:\n")
                    for dif in diferencias:
                        txt_file.write(f"  Pos {dif['posicion']}: Arch1[{dif['arch1']}] vs Arch2[{dif['arch2']}]\n")
                else:
                    resultado['resultado'] = 'no_diferencias'
                    txt_file.write(f"Línea {i+1}: [NO HAY DIFERENCIAS]\n")
            
            json.dump(resultado, json_file)
            json_file.write('\n')
            
            if len(muestra_resultados) < 20:
                muestra_resultados.append(resultado)
            
            pbar.update(1)
    
    # Manejar líneas sobrantes
    def manejar_sobrantes(lineas, inicio, nombre_archivo):
        for i in range(inicio, len(lineas)):
            contenido = lineas[i][:50] + '...' if len(lineas[i]) > 50 else lineas[i]
            txt_file.write(f"Línea {i+1}: [SOLO EN {nombre_archivo.upper()}: {contenido}]\n")
            json.dump({
                'linea': i+1,
                'resultado': f'solo_en_{nombre_archivo}',
                'contenido': lineas[i]
            }, json_file)
            json_file.write('\n')
    
    if len(lineas1) > len(lineas2):
        manejar_sobrantes(lineas1, len(lineas2), "archivo1")
    elif len(lineas2) > len(lineas1):
        manejar_sobrantes(lineas2, len(lineas1), "archivo2")
    
    json_file.close()
    txt_file.close()
    
    return muestra_resultados, total_diferencias

def main():
    ruta_archivo1 = "chain-one-test.fna"
    ruta_archivo2 = "chain-two-test.fna"
    
    print("\nIniciando comparación con GPU...")
    start_time = time.time()
    
    cpu_temp, ram_usage, gpu_name = get_system_info()
    print(f"\nGPU: {gpu_name} | Uso RAM: {ram_usage}% | Temp CPU: {cpu_temp}°C")
    
    try:
        muestra, total = procesar_archivos_gpu(ruta_archivo1, ruta_archivo2)
        
        tiempo_total = time.time() - start_time
        print(f"\nProcesamiento completado en {tiempo_total:.2f} segundos")
        print(f"Total de diferencias encontradas: {total}")
        
        print("\nMuestra de resultados:")
        for res in muestra[:20]:
            if res.get('resultado') == 'identical':
                print(f"Línea {res['linea']}: [CADENA IDÉNTICA]")
            elif 'diferencias' in res:
                print(f"Línea {res['linea']}:")
                for dif in res['diferencias'][:5]:  # Mostrar solo primeras 5 diferencias
                    print(f"  Pos {dif['posicion']}: Arch1[{dif['arch1']}] vs Arch2[{dif['arch2']}]")
                if len(res['diferencias']) > 5:
                    print(f"  ... y {len(res['diferencias'])-5} diferencias más")
            else:
                print(f"Línea {res['linea']}: [NO HAY DIFERENCIAS]")
        
        print("\nResultados guardados en:")
        print("- diferencias_gpu.json")
        print("- diferencias_gpu.txt")
    
    except Exception as e:
        print(f"\nError durante el procesamiento: {str(e)}")

if __name__ == '__main__':
    main()