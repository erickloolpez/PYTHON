import time
import numpy as np
import pyopencl as cl
from tqdm import tqdm
import os

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
    ctx = cl.create_some_context()
    devices = ctx.devices
    gpu_name = devices[0].name if devices else "No GPU detectada"
    
    return cpu_temp, ram_usage, gpu_name

def procesar_lote_gpu(lineas_lote, numeros_linea, ctx, queue, max_len):
    """Procesa un lote de líneas en la GPU y devuelve resultados"""
    # Preparar datos para GPU (rellenar líneas si es necesario)
    lineas_padded = [linea.ljust(max_len) for linea in lineas_lote]
    lineas_array = np.array([list(linea) for linea in lineas_padded], dtype='S1')
    
    # Buffers para GPU
    mf = cl.mem_flags
    lineas_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lineas_array)
    resultados_buf = cl.Buffer(ctx, mf.WRITE_ONLY, lineas_array.shape[0] * 2 * np.dtype(np.int32).itemsize)
    
    # Kernel de OpenCL (el mismo que antes)
    kernel_code = """
    __kernel void encontrar_minimo(__global char* lineas, int max_len, __global int* resultados) {
        int idx = get_global_id(0);
        char min_char = 127;
        int min_pos = 0;
        
        for (int i = 0; i < max_len; i++) {
            char c = lineas[idx * max_len + i];
            if (c != ' ' && c < min_char) {
                min_char = c;
                min_pos = i + 1;
            }
        }
        
        resultados[idx * 2] = min_char;
        resultados[idx * 2 + 1] = min_pos;
    }
    """
    
    # Compilar y ejecutar
    prg = cl.Program(ctx, kernel_code).build()
    prg.encontrar_minimo(queue, (len(lineas_lote),), None, lineas_buf, np.int32(max_len), resultados_buf)
    
    # Recuperar resultados
    resultados_gpu = np.empty(len(lineas_lote) * 2, dtype=np.int32)
    cl.enqueue_copy(queue, resultados_gpu, resultados_buf).wait()
    
    # Formatear resultados
    resultados_lote = []
    for i in range(len(lineas_lote)):
        num_linea = numeros_linea[i]
        min_char = chr(resultados_gpu[i * 2])
        min_pos = resultados_gpu[i * 2 + 1]
        resultados_lote.append((num_linea, min_pos, min_char))
    
    # Liberar buffers de GPU explícitamente
    lineas_buf.release()
    resultados_buf.release()
    
    return resultados_lote

def procesar_archivo_por_lotes_gpu(ruta_archivo, batch_size=10000):
    """Procesa el archivo por lotes usando GPU"""
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    # Contar líneas totales para la barra de progreso
    with open(ruta_archivo, 'r') as f:
        total_lineas = sum(1 for _ in f)
    
    # Reabrir el archivo para procesar por lotes
    resultados_totales = []
    lineas_lote = []
    numeros_linea_lote = []
    max_len = 0  # Longitud máxima de línea en el lote actual
    
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
                resultados_totales.extend(resultados_lote)
                lineas_lote = []
                numeros_linea_lote = []
                max_len = 0
        
        # Procesar el último lote (si queda algo)
        if lineas_lote:
            resultados_lote = procesar_lote_gpu(lineas_lote, numeros_linea_lote, ctx, queue, max_len)
            resultados_totales.extend(resultados_lote)
    
    return resultados_totales

def main():
    ruta_archivo = "genoma.fna"
    batch_size = 10000  # Ajusta según tu RAM/GPU
    
    print("\nIniciando procesamiento por lotes con GPU...")
    start_time = time.time()
    
    # Info del sistema
    cpu_temp, ram_usage, gpu_name = get_system_info()
    print(f"GPU: {gpu_name} | RAM Usage: {ram_usage}% | Batch Size: {batch_size} líneas")
    
    # Procesar
    resultados = procesar_archivo_por_lotes_gpu(ruta_archivo, batch_size)
    
    # Resultados
    tiempo_total = time.time() - start_time
    print(f"\nProcesado en {tiempo_total:.2f} segundos")
    print(f"Total líneas procesadas: {len(resultados)}")
    
    # Guardar (opcional: también puedes hacerlo por lotes)
    with open("resultados_gpu_lotes.txt", "w") as f:
        f.write("Línea\tPosición\tCarácter\n")
        for linea, pos, char in resultados:
            f.write(f"{linea}\t{pos}\t{char}\n")
    
    print("\nResultados guardados en 'resultados_gpu_lotes.txt'")

if __name__ == '__main__':
    main()