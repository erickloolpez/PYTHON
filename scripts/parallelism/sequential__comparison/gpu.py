import time
import numpy as np
import pyopencl as cl
from tqdm import tqdm
import os
import json

def get_system_info():
    """Obtiene información del sistema (GPU, RAM, CPU temp)"""
    try:
        import psutil
        temp = psutil.sensors_temperatures()
        cpu_temp = temp['coretemp'][0].current if 'coretemp' in temp else 'N/A'
    except Exception as e:
        cpu_temp = 'N/A'
    
    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    
    # Crear contexto OpenCL
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()
    gpu_name = devices[0].name if devices else "No GPU detectada"
    
    return cpu_temp, ram_usage, gpu_name

def procesar_lote_gpu(lineas_lote, numeros_linea, ctx, queue, max_len):
    """Procesa un lote de líneas en la GPU para encontrar coincidencias consecutivas"""
    if len(lineas_lote) < 2:
        return []
    
    # Preparar datos para GPU (rellenar líneas si es necesario)
    lineas_padded = [linea.ljust(max_len) for linea in lineas_lote]
    lineas_array = np.array([list(linea) for linea in lineas_padded], dtype='S1')
    
    # Buffers para GPU
    mf = cl.mem_flags
    lineas_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lineas_array)
    resultados_buf = cl.Buffer(ctx, mf.WRITE_ONLY, (len(lineas_lote)-1) * max_len * np.dtype(np.int8).itemsize)
    
    # Kernel de OpenCL optimizado para comparación de líneas consecutivas
    kernel_code = """
    __kernel void encontrar_coincidencias(
        __global char* lineas, 
        int max_len, 
        int num_lineas, 
        __global char* resultados
    ) {
        int idx = get_global_id(0);
        
        // Solo comparar si no es la última línea
        if (idx < num_lineas - 1) {
            for (int pos = 0; pos < max_len; pos++) {
                char actual = lineas[idx * max_len + pos];
                char siguiente = lineas[(idx + 1) * max_len + pos];
                
                // Marcar con 1 si hay coincidencia, 0 si no
                resultados[idx * max_len + pos] = (actual == siguiente && actual != ' ') ? actual : 0;
            }
        }
    }
    """
    
    # Compilar y ejecutar
    prg = cl.Program(ctx, kernel_code).build()
    prg.encontrar_coincidencias(
        queue, 
        (len(lineas_lote)-1,), 
        None, 
        lineas_buf, 
        np.int32(max_len), 
        np.int32(len(lineas_lote)), 
        resultados_buf
    )
    
    # Recuperar resultados
    resultados_gpu = np.empty((len(lineas_lote)-1) * max_len, dtype=np.int8)
    cl.enqueue_copy(queue, resultados_gpu, resultados_buf).wait()
    
    # Formatear resultados
    resultados_lote = []
    for i in range(len(lineas_lote)-1):
        coincidencias = {}
        for pos in range(max_len):
            char = chr(resultados_gpu[i * max_len + pos])
            if char != '\x00':  # Solo caracteres con coincidencia
                coincidencias[pos+1] = char  # +1 para posición basada en 1
        
        if coincidencias:
            linea1_num = numeros_linea[i]
            linea2_num = numeros_linea[i+1]
            resultados_lote.append((linea1_num, linea2_num, coincidencias))
    
    # Liberar buffers de GPU explícitamente
    lineas_buf.release()
    resultados_buf.release()
    
    return resultados_lote

def procesar_archivo_por_lotes_gpu(ruta_archivo, batch_size=10000):
    """Procesa el archivo por lotes usando GPU para comparación de líneas consecutivas"""
    # Configuración OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    # Contar líneas totales para la barra de progreso
    with open(ruta_archivo, 'r') as f:
        total_lineas = sum(1 for _ in f)
    
    # Archivos de salida
    json_file = open("coincidencias_gpu.json", "w")
    txt_file = open("coincidencias_gpu.txt", "w")
    txt_file.write("Resultados de comparación de líneas consecutivas (GPU):\n")
    txt_file.write("Formato: Línea X y Y -> Coinciden en las posiciones: [pos:char,...]\n\n")
    
    # Variables para procesamiento por lotes
    lineas_lote = []
    numeros_linea_lote = []
    max_len = 0
    total_coincidencias = 0
    muestra_resultados = []
    
    with open(ruta_archivo, 'r') as f, tqdm(total=total_lineas, desc="Procesando") as pbar:
        for num_linea, linea in enumerate(f, 1):
            linea = linea.strip()
            if linea and not linea.startswith('>'):
                lineas_lote.append(linea)
                numeros_linea_lote.append(num_linea)
                max_len = max(max_len, len(linea))
            
            pbar.update(1)
            
            # Procesar lote cuando alcance el tamaño definido
            if len(lineas_lote) >= batch_size:
                resultados_lote = procesar_lote_gpu(lineas_lote, numeros_linea_lote, ctx, queue, max_len)
                total_coincidencias += len(resultados_lote)
                
                # Guardar resultados en archivos
                for res in resultados_lote:
                    json.dump(res, json_file)
                    json_file.write('\n')
                    
                    linea1, linea2, coincidencias = res
                    txt_line = f"Línea {linea1} y {linea2} -> Coinciden en las posiciones: ["
                    txt_line += ','.join(f"{pos}:{char}" for pos, char in sorted(coincidencias.items()))
                    txt_line += "]\n"
                    txt_file.write(txt_line)
                
                # Mantener muestra para mostrar al final
                if len(muestra_resultados) < 20:
                    muestra_resultados.extend(resultados_lote[:20-len(muestra_resultados)])
                
                # Reiniciar lote
                lineas_lote = []
                numeros_linea_lote = []
                max_len = 0
        
        # Procesar el último lote (si queda algo)
        if lineas_lote:
            resultados_lote = procesar_lote_gpu(lineas_lote, numeros_linea_lote, ctx, queue, max_len)
            total_coincidencias += len(resultados_lote)
            
            # Guardar resultados finales
            for res in resultados_lote:
                json.dump(res, json_file)
                json_file.write('\n')
                
                linea1, linea2, coincidencias = res
                txt_line = f"Línea {linea1} y {linea2} -> Coinciden en las posiciones: ["
                txt_line += ','.join(f"{pos}:{char}" for pos, char in sorted(coincidencias.items()))
                txt_line += "]\n"
                txt_file.write(txt_line)
            
            if len(muestra_resultados) < 20:
                muestra_resultados.extend(resultados_lote[:20-len(muestra_resultados)])
    
    json_file.close()
    txt_file.close()
    
    return muestra_resultados, total_coincidencias

def main():
    ruta_archivo = "genoma.fna"
    batch_size = 10000  # Ajusta según tu RAM/GPU
    
    print("\nIniciando procesamiento por lotes con GPU...")
    start_time = time.time()
    
    # Info del sistema
    cpu_temp, ram_usage, gpu_name = get_system_info()
    print(f"GPU: {gpu_name} | RAM Usage: {ram_usage}% | Batch Size: {batch_size} líneas")
    
    # Procesar
    muestra_resultados, total_coincidencias = procesar_archivo_por_lotes_gpu(ruta_archivo, batch_size)
    
    # Resultados
    tiempo_total = time.time() - start_time
    print(f"\nProcesado en {tiempo_total:.2f} segundos")
    print(f"Total de pares con coincidencias encontrados: {total_coincidencias}")
    
    # Mostrar muestra de resultados
    print("\nMuestra de resultados:")
    for i, (linea1, linea2, coincidencias) in enumerate(muestra_resultados[:20]):
        print(f"Línea {linea1} y {linea2} -> Coinciden en las posiciones: [", end="")
        print(','.join(f"{pos}:{char}" for pos, char in sorted(coincidencias.items())), end="")
        print("]")
    
    print("\nResultados completos guardados en:")
    print("- 'coincidencias_gpu.json' (formato estructurado)")
    print("- 'coincidencias_gpu.txt' (formato legible)")

if __name__ == '__main__':
    main()