import time
import numpy as np
import pyopencl as cl
from tqdm import tqdm
import os
import json

def get_system_info():
    """Obtiene información del sistema (GPU, RAM) de forma segura"""
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        ram_usage = memory_info.percent

        try:
            temp = psutil.sensors_temperatures()
            if temp:
                cpu_temp = next(iter(temp.values()))[0].current
            else:
                cpu_temp = 'N/A'
        except Exception:
            cpu_temp = 'N/A'
    except ImportError:
        cpu_temp = 'N/A'
        ram_usage = 'N/A'

    try:
        platforms = cl.get_platforms()
        gpu_devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
        if not gpu_devices:
            raise RuntimeError("No se encontraron dispositivos GPU.")
        ctx = cl.Context(devices=gpu_devices)
        gpu_name = gpu_devices[0].name
    except Exception as e:
        gpu_name = f"Error: {e}"

    return cpu_temp, ram_usage, gpu_name

def preparar_datos(lineas1, lineas2, max_len):
    """Prepara las líneas para enviar a la GPU (relleno y codificación)"""
    l1 = [l.ljust(max_len) for l in lineas1]
    l2 = [l.ljust(max_len) for l in lineas2]
    a1 = np.array([list(x.encode('utf-8')) for x in l1], dtype='uint8')
    a2 = np.array([list(x.encode('utf-8')) for x in l2], dtype='uint8')
    return a1, a2

def comparar_en_gpu(lineas1, lineas2, ctx, queue, max_len):
    """Envía los datos a la GPU y devuelve diferencias"""
    mf = cl.mem_flags

    datos1, datos2 = preparar_datos(lineas1, lineas2, max_len)

    buf1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=datos1)
    buf2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=datos2)
    diffs = np.empty((len(lineas1), max_len), dtype='uint8')
    diffs_buf = cl.Buffer(ctx, mf.WRITE_ONLY, diffs.nbytes)

    kernel_code = """
    __kernel void comparar_lineas(
        __global const uchar* datos1,
        __global const uchar* datos2,
        __global uchar* diffs,
        const int max_len
    ) {
        int i = get_global_id(0);
        for (int j = 0; j < max_len; j++) {
            uchar c1 = datos1[i * max_len + j];
            uchar c2 = datos2[i * max_len + j];
            diffs[i * max_len + j] = (c1 == c2) ? 0 : 1;
        }
    }
    """

    prg = cl.Program(ctx, kernel_code).build()
    prg.comparar_lineas(
        queue,
        (len(lineas1),),
        None,
        buf1, buf2, diffs_buf, np.int32(max_len)
    )

    cl.enqueue_copy(queue, diffs, diffs_buf).wait()

    buf1.release()
    buf2.release()
    diffs_buf.release()

    return diffs

def procesar_comparacion_gpu(ruta1, ruta2, batch_size=10000):
    """Procesa la comparación entre dos archivos usando la GPU"""
    platforms = cl.get_platforms()
    gpu_devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=gpu_devices)
    queue = cl.CommandQueue(ctx)

    def lineas_validas(f):
        return [l.strip() for l in f if l.strip() and not l.startswith('>')]

    with open(ruta1, 'r') as f1, open(ruta2, 'r') as f2:
        l1 = lineas_validas(f1)
        l2 = lineas_validas(f2)

    total = min(len(l1), len(l2))
    resultados = []

    with tqdm(total=total, desc="Comparando") as pbar:
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            bloque1 = l1[i:end]
            bloque2 = l2[i:end]
            max_len = max(max(map(len, bloque1)), max(map(len, bloque2)))
            diffs = comparar_en_gpu(bloque1, bloque2, ctx, queue, max_len)

            for j in range(end - i):
                num_linea = i + j + 1
                linea_diff = diffs[j]
                if np.sum(linea_diff) == 0:
                    resultados.append({'linea': num_linea, 'resultado': 'identical'})
                else:
                    detalles = []
                    for pos, val in enumerate(linea_diff):
                        if val != 0:
                            c1 = bloque1[j][pos] if pos < len(bloque1[j]) else ''
                            c2 = bloque2[j][pos] if pos < len(bloque2[j]) else ''
                            detalles.append({'posicion': pos + 1, 'cadena1': c1, 'cadena2': c2})
                    resultados.append({'linea': num_linea, 'diferencias': detalles})
            pbar.update(end - i)

    return resultados, len(resultados)

def main():
    archivo1 = "../chain-one.fna"
    archivo2 = "../chain-two.fna"
    batch_size = 10000

    print("Iniciando comparación GPU...")
    start = time.time()

    cpu_temp, ram_usage, gpu_name = get_system_info()
    print(f"GPU: {gpu_name} | RAM: {ram_usage}% | CPU Temp: {cpu_temp}°C")

    resultados, total = procesar_comparacion_gpu(archivo1, archivo2, batch_size)

    tiempo = time.time() - start
    print(f"\nTotal de diferencias encontradas: {total}")
    print(f"Tiempo total: {tiempo:.2f} segundos")

    with open("diferencias_gpu.json", "w") as jsonfile, open("diferencias_gpu.txt", "w") as txtfile:
        for res in resultados:
            json.dump(res, jsonfile)
            jsonfile.write('\n')

            if res.get('resultado') == 'identical':
                txtfile.write(f"Línea {res['linea']}: [IDÉNTICA]\n")
            else:
                txtfile.write(f"Línea {res['linea']}:\n")
                for dif in res['diferencias']:
                    txtfile.write(f"  Pos {dif['posicion']}: Arch1[{dif['cadena1']}] vs Arch2[{dif['cadena2']}]\n")

    print("\nResultados guardados en:")
    print("- diferencias_gpu.json")
    print("- diferencias_gpu.txt")

if __name__ == '__main__':
    main()
