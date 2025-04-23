from multiprocessing import Process, Queue, cpu_count
from itertools import islice
from tqdm import tqdm
import json
import os
import time
import psutil

MAX_MEMORY_USAGE = 0.8
BUFFER_SIZE = 50

def get_available_memory():
    return psutil.virtual_memory().available

def leer_bloque_validas(f, n):
    """Lee n líneas válidas desde un archivo"""
    bloque = []
    while len(bloque) < n:
        try:
            linea = next(f).strip()
            if linea and not linea.startswith('>'):
                bloque.append(linea)
        except StopIteration:
            break
    return bloque

def comparar_lineas(linea1, linea2, num_linea):
    if linea1 == linea2:
        return {'linea': num_linea, 'resultado': 'identical'}
    
    diferencias = []
    len_min = min(len(linea1), len(linea2))
    
    for pos in range(len_min):
        if linea1[pos] != linea2[pos]:
            diferencias.append({
                'posicion': pos+1,
                'cadena1': linea1[pos],
                'cadena2': linea2[pos]
            })
    
    if len(linea1) != len(linea2):
        extra = range(len_min, max(len(linea1), len(linea2)))
        for pos in extra:
            diferencias.append({
                'posicion': pos+1,
                'cadena1': linea1[pos] if pos < len(linea1) else None,
                'cadena2': linea2[pos] if pos < len(linea2) else None
            })
    
    return {'linea': num_linea, 'diferencias': diferencias}

def worker(bloque1, bloque2, start_idx, result_queue):
    resultados = []
    for i, (l1, l2) in enumerate(zip(bloque1, bloque2)):
        res = comparar_lineas(l1, l2, start_idx + i + 1)
        resultados.append(res)
    result_queue.put(resultados)

def escribir_resultados(resultados, json_file, txt_file):
    for res in resultados:
        json.dump(res, json_file)
        json_file.write('\n')
        if res.get('resultado') == 'identical':
            txt_file.write(f"Línea {res['linea']}: [CADENA IDÉNTICA]\n")
        else:
            txt_file.write(f"Línea {res['linea']}:\n")
            for dif in res.get('diferencias', []):
                txt_file.write(f"  Pos {dif['posicion']}: Arch1[{dif['cadena1']}] vs Arch2[{dif['cadena2']}]\n")

def procesar_archivos(ruta1, ruta2):
    muestra_resultados = []
    total_diferencias = 0
    buffer_resultados = []

    available_mem = get_available_memory()
    mem_por_linea = 1000  # estimación conservadora
    max_lineas_en_mem = int((available_mem * MAX_MEMORY_USAGE) / mem_por_linea)
    block_size = max(min(max_lineas_en_mem // cpu_count(), 10000), 1)

    tiempo_inicio = time.time()

    with open(ruta1, 'r') as f1, open(ruta2, 'r') as f2, \
         open("diferencias.json", "w") as json_file, \
         open("diferencias.txt", "w") as txt_file:

        txt_file.write("Comparación línea por línea:\n")
        txt_file.write(f"Archivo 1: {ruta1}\n")
        txt_file.write(f"Archivo 2: {ruta2}\n\n")

        linea_global = 0
        with tqdm(desc="Procesando") as pbar:
            while True:
                bloque1 = leer_bloque_validas(f1, block_size)
                bloque2 = leer_bloque_validas(f2, block_size)
                tamano_bloque = max(len(bloque1), len(bloque2))

                if not bloque1 and not bloque2:
                    break

                # Rellenar el bloque más corto con cadenas vacías para igualar longitud
                while len(bloque1) < tamano_bloque:
                    bloque1.append('')
                while len(bloque2) < tamano_bloque:
                    bloque2.append('')

                num_workers = min(cpu_count(), tamano_bloque)
                lines_per_worker = tamano_bloque // num_workers
                processes = []
                result_queue = Queue()

                for i in range(num_workers):
                    inicio = i * lines_per_worker
                    fin = (i + 1) * lines_per_worker if i < num_workers - 1 else tamano_bloque
                    p = Process(
                        target=worker,
                        args=(bloque1[inicio:fin], bloque2[inicio:fin], linea_global + inicio, result_queue)
                    )
                    p.start()
                    processes.append(p)

                for _ in processes:
                    resultados = result_queue.get()
                    buffer_resultados.extend(resultados)

                    if len(buffer_resultados) >= BUFFER_SIZE:
                        escribir_resultados(buffer_resultados, json_file, txt_file)
                        total_diferencias += len(buffer_resultados)
                        if len(muestra_resultados) < 20:
                            muestra_resultados.extend(
                                [r for r in buffer_resultados if 'diferencias' in r][:20 - len(muestra_resultados)]
                            )
                        buffer_resultados = []

                for p in processes:
                    p.join()

                linea_global += tamano_bloque
                pbar.update(tamano_bloque)

        # Escribir el buffer restante
        if buffer_resultados:
            escribir_resultados(buffer_resultados, json_file, txt_file)
            total_diferencias += len(buffer_resultados)

    tiempo_total = time.time() - tiempo_inicio
    return muestra_resultados, total_diferencias, tiempo_total

def main():
    ruta1 = "../chain-one.fna"
    ruta2 = "../chain-two.fna"

    print("Iniciando comparación optimizada...")
    muestra, total, tiempo = procesar_archivos(ruta1, ruta2)

    print(f"\nComparación finalizada en {tiempo:.2f} segundos")
    print(f"Diferencias totales encontradas: {total}\n")

    print("Muestra de diferencias:")
    for res in muestra:
        print(f"Línea {res['linea']}:")
        for dif in res.get('diferencias', [])[:5]:
            print(f"  Pos {dif['posicion']}: Arch1[{dif['cadena1']}] vs Arch2[{dif['cadena2']}]")
        if len(res.get('diferencias', [])) > 5:
            print(f"  ... y {len(res['diferencias'])-5} más")

    print("\nResultados guardados en:")
    print("- diferencias.json")
    print("- diferencias.txt")

if __name__ == '__main__':
    main()
