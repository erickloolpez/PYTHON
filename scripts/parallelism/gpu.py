import time
import numpy as np  # Esta es la línea que faltaba
import pandas as pd
import psutil
import pyopencl as cl  # Biblioteca para OpenCL

def get_system_info():
    try:
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

def compare_genotypes_gpu(df):
    # Configurar OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    # Preparar datos
    genotype1 = df['genotype_1'].values.astype('S10')  # Asume strings de hasta 10 chars
    genotype2 = df['genotype_2'].values.astype('S10')
    
    # Crear buffers en GPU
    mf = cl.mem_flags
    g1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=genotype1)
    g2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=genotype2)
    result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, genotype1.nbytes)
    
    # Código del kernel OpenCL
    kernel_code = """
    __kernel void compare_genotypes(__global char* g1, __global char* g2, __global int* result) {
        int idx = get_global_id(0);
        result[idx] = (g1[idx] == g2[idx]) ? 1 : 0;
    }
    """
    
    # Compilar y ejecutar
    prg = cl.Program(ctx, kernel_code).build()
    prg.compare_genotypes(queue, genotype1.shape, None, g1_buf, g2_buf, result_buf)
    
    # Obtener resultados
    result = np.empty_like(genotype1, dtype=np.int32)
    cl.enqueue_copy(queue, result, result_buf)
    
    matches = np.sum(result)
    differences = len(df) - matches
    
    return matches, differences

if __name__ == '__main__':
    # Read the file, skipping comment lines starting with #
    chainOne = pd.read_csv("ManuSporny-genome_CP00.txt", 
                          comment="#",       # skip lines that start with #
                          delim_whitespace=True,  # split on any whitespace
                          names=["rsid", "chromosome", "position", "genotype"])  # manually set column names

    chainTwo = pd.read_csv("ManuSporny-genome_CP01.txt", 
                          comment="#",       # skip lines that start with #
                          delim_whitespace=True,  # split on any whitespace
                          names=["rsid", "chromosome", "position", "genotype"])  # manually set column names

    # Merge on 'rsid' to align SNPs
    merged = pd.merge(chainOne[['rsid', 'genotype']], chainTwo[['rsid', 'genotype']], on='rsid', suffixes=('_1', '_2'))

    print("\nEjecutando comparación en GPU (OpenCL)...\n")
    start_time = time.time()
    
    cpu_temp, ram_usage, gpu_name = get_system_info()
    print(f"Información del sistema - CPU Temp: {cpu_temp}°C, RAM Usage: {ram_usage}%")
    print(f"Dispositivo OpenCL: {gpu_name}")
    
    matches, differences = compare_genotypes_gpu(merged)
    
    # Calculate percentage of difference
    total = len(merged)
    difference_percent = (differences / total) * 100

    print(f"Total SNPs comparados: {total}")
    print(f"Genotipos coincidentes: {matches}")
    print(f"Genotipos diferentes: {differences}")
    print(f"Porcentaje de diferencia: {difference_percent:.2f}%")

    # Calculate execution time
    end_time = time.time()  
    execution_time = end_time - start_time  

    print(f"Tiempo de ejecución con GPU: {execution_time:.2f} segundos")