from multiprocessing import Process, Queue, cpu_count, Manager
import time
from tqdm import tqdm
import json

def leer_lineas_validas(ruta_archivo):
    """Lee solo líneas válidas (no vacías y no encabezados) de un archivo"""
    lineas = []
    try:
        with open(ruta_archivo, 'r') as f:
            for linea in f:
                linea = linea.strip()
                if linea and not linea.startswith('>'):
                    lineas.append(linea)
        return lineas
    except Exception as e:
        print(f"Error leyendo {ruta_archivo}: {str(e)}")
        return None

def comparar_lote_lineas(lote1, lote2, inicio_linea, resultado_queue):
    """Compara un lote de líneas entre los dos archivos"""
    resultados_lote = []
    
    for i in range(min(len(lote1), len(lote2))):
        linea1 = lote1[i]
        linea2 = lote2[i]
        linea_num = inicio_linea + i + 1
        
        if linea1 == linea2:
            resultados_lote.append({
                'linea': linea_num,
                'resultado': 'identical'
            })
            continue
            
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
            if len(linea1) > len(linea2):
                for pos in range(len(linea2), len(linea1)):
                    diferencias.append({
                        'posicion': pos+1,
                        'cadena1': linea1[pos],
                        'cadena2': None
                    })
            else:
                for pos in range(len(linea1), len(linea2)):
                    diferencias.append({
                        'posicion': pos+1,
                        'cadena1': None,
                        'cadena2': linea2[pos]
                    })
        
        resultados_lote.append({
            'linea': linea_num,
            'diferencias': diferencias
        })
    
    resultado_queue.put(resultados_lote)

def procesar_archivos_paralelo(ruta_archivo1, ruta_archivo2, num_procesos, tamano_lote=1000):
    """Procesa los archivos en paralelo por lotes"""
    lineas1 = leer_lineas_validas(ruta_archivo1)
    lineas2 = leer_lineas_validas(ruta_archivo2)
    
    if lineas1 is None or lineas2 is None:
        return [], 0, 0
    
    total_lineas = max(len(lineas1), len(lineas2))
    print(f"Total de líneas a comparar: {total_lineas}")
    
    try:
        json_file = open("diferencias_lineas.json", "w")
        txt_file = open("diferencias_lineas.txt", "w")
        txt_file.write("Comparación línea por línea entre archivos:\n")
        txt_file.write(f"Archivo 1: {ruta_archivo1}\n")
        txt_file.write(f"Archivo 2: {ruta_archivo2}\n\n")
    except Exception as e:
        print(f"Error creando archivos de salida: {str(e)}")
        return [], 0, 0
    
    manager = Manager()
    resultado_queue = manager.Queue()
    procesos = []
    tiempo_inicio = time.perf_counter()
    total_coincidencias = 0
    muestra_resultados = []
    
    try:
        with tqdm(total=total_lineas, desc="Procesando líneas") as pbar:
            lote_actual1 = []
            lote_actual2 = []
            linea_actual = 0
            
            while linea_actual < total_lineas:
                if linea_actual < len(lineas1):
                    lote_actual1.append(lineas1[linea_actual])
                if linea_actual < len(lineas2):
                    lote_actual2.append(lineas2[linea_actual])
                
                linea_actual += 1
                pbar.update(1)
                
                if len(lote_actual1) + len(lote_actual2) >= tamano_lote or linea_actual == total_lineas:
                    if lote_actual1 or lote_actual2:
                        while len(procesos) >= num_procesos:
                            for i, p in enumerate(procesos):
                                if not p.is_alive():
                                    procesos.pop(i)
                                    while not resultado_queue.empty():
                                        resultados = resultado_queue.get()
                                        total_coincidencias += len(resultados)
                                        for res in resultados:
                                            json.dump(res, json_file)
                                            json_file.write('\n')
                                            
                                            if 'resultado' in res and res['resultado'] == 'identical':
                                                txt_file.write(f"Línea {res['linea']} -> La cadena es idéntica\n")
                                            else:
                                                txt_file.write(f"Línea {res['linea']} -> Diferencias:\n")
                                                for dif in res.get('diferencias', []):
                                                    txt_file.write(f"  Pos {dif['posicion']}: ")
                                                    txt_file.write(f"Arch1[{dif['cadena1']}] vs Arch2[{dif['cadena2']}]\n")
                                        if len(muestra_resultados) < 20:
                                            muestra_resultados.extend(resultados[:20-len(muestra_resultados)])
                                    break
                            else:
                                time.sleep(0.1)
                        
                        inicio_global = linea_actual - len(lote_actual1) - len(lote_actual2)
                        p = Process(target=comparar_lote_lineas,
                                  args=(lote_actual1, lote_actual2, inicio_global, resultado_queue))
                        p.start()
                        procesos.append(p)
                        lote_actual1 = []
                        lote_actual2 = []
        
        for p in procesos:
            p.join()
            while not resultado_queue.empty():
                resultados = resultado_queue.get()
                total_coincidencias += len(resultados)
                for res in resultados:
                    json.dump(res, json_file)
                    json_file.write('\n')
                    
                    if 'resultado' in res and res['resultado'] == 'identical':
                        txt_file.write(f"Línea {res['linea']} -> La cadena es idéntica\n")
                    else:
                        txt_file.write(f"Línea {res['linea']} -> Diferencias:\n")
                        for dif in res.get('diferencias', []):
                            txt_file.write(f"  Pos {dif['posicion']}: ")
                            txt_file.write(f"Arch1[{dif['cadena1']}] vs Arch2[{dif['cadena2']}]\n")
                if len(muestra_resultados) < 20:
                    muestra_resultados.extend(resultados[:20-len(muestra_resultados)])
        
        # Manejar líneas sobrantes
        if len(lineas1) > len(lineas2):
            for i in range(len(lineas2), len(lineas1)):
                txt_file.write(f"Línea {i+1} -> Solo existe en el primer archivo: {lineas1[i][:50]}...\n")
                json.dump({
                    'linea': i+1,
                    'resultado': 'solo_en_archivo1',
                    'contenido': lineas1[i]
                }, json_file)
                json_file.write('\n')
        elif len(lineas2) > len(lineas1):
            for i in range(len(lineas1), len(lineas2)):
                txt_file.write(f"Línea {i+1} -> Solo existe en el segundo archivo: {lineas2[i][:50]}...\n")
                json.dump({
                    'linea': i+1,
                    'resultado': 'solo_en_archivo2',
                    'contenido': lineas2[i]
                }, json_file)
                json_file.write('\n')
        
        tiempo_total = time.perf_counter() - tiempo_inicio
        
    except Exception as e:
        print(f"Error durante el procesamiento: {str(e)}")
        tiempo_total = time.perf_counter() - tiempo_inicio
        return muestra_resultados, total_coincidencias, tiempo_total
    
    finally:
        json_file.close()
        txt_file.close()
    
    return muestra_resultados, total_coincidencias, tiempo_total

def main():
    ruta_archivo1 = "chain-one-test.fna"
    ruta_archivo2 = "chain-two-test.fna"
    num_procesos = min(cpu_count(), 8)
    tamano_lote = 1000
    
    print(f"Iniciando comparación línea por línea entre:")
    print(f"Archivo 1: {ruta_archivo1}")
    print(f"Archivo 2: {ruta_archivo2}")
    
    muestra, total, tiempo = procesar_archivos_paralelo(
        ruta_archivo1, ruta_archivo2, num_procesos, tamano_lote)
    
    print(f"\nProcesamiento completado en {tiempo:.2f} segundos")
    print(f"Total de líneas comparadas: {total}")
    
    print("\nMuestra de resultados:")
    for res in muestra[:20]:
        if 'resultado' in res and res['resultado'] == 'identical':
            print(f"Línea {res['linea']}: [CADENA IDÉNTICA]")
        else:
            print(f"Línea {res['linea']}:")
            for dif in res.get('diferencias', [])[:5]:
                print(f"  Pos {dif['posicion']}: Arch1[{dif['cadena1']}] vs Arch2[{dif['cadena2']}]")
            if len(res.get('diferencias', [])) > 5:
                print(f"  ... y {len(res['diferencias'])-5} diferencias más")

if __name__ == '__main__':
    main()