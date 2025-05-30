from multiprocessing import Process, Queue, cpu_count, Manager
import time
from tqdm import tqdm
import os
import json

def comparar_lineas(lote_lineas, inicio_linea_global, resultado_queue):
    """
    Compara líneas consecutivas y encuentra caracteres iguales en la misma posición
    Devuelve los resultados en formato: (linea1, linea2, {pos:char})
    """
    resultados_lote = []
    
    if len(lote_lineas) < 2:
        return
        
    for i in range(len(lote_lineas)-1):
        linea_actual = lote_lineas[i].strip()
        linea_siguiente = lote_lineas[i+1].strip()
        
        # Saltar líneas vacías o de encabezado
        if (not linea_actual or linea_actual.startswith('>') or 
            not linea_siguiente or linea_siguiente.startswith('>')):
            continue
            
        # Determinar la longitud mínima para comparar
        longitud_comparacion = min(len(linea_actual), len(linea_siguiente))
        coincidencias = {}
        
        # Comparar caracteres en cada posición
        for pos in range(longitud_comparacion):
            if linea_actual[pos] == linea_siguiente[pos]:
                # Usamos pos+1 para mostrar posición basada en 1
                coincidencias[pos+1] = linea_actual[pos]
        
        if coincidencias:
            linea1_num = inicio_linea_global + i + 1  # +1 porque las líneas empiezan en 1
            linea2_num = linea1_num + 1
            resultados_lote.append((linea1_num, linea2_num, coincidencias))
    
    resultado_queue.put(resultados_lote)

def procesar_archivo_por_lotes(ruta_archivo, num_procesos, tamano_lote=10000):
    """
    Procesa el archivo por lotes usando múltiples procesos
    """
    # Contar líneas totales
    with open(ruta_archivo, 'r') as f:
        total_lineas = sum(1 for _ in f)
    print(f"Total de líneas a procesar: {total_lineas}")
    
    # Archivos de salida
    json_file = open("coincidencias_consecutivas.json", "w")
    txt_file = open("coincidencias_consecutivas.txt", "w")
    txt_file.write("Resultados de comparación de líneas consecutivas:\n")
    txt_file.write("Formato: Línea X y Y -> Coinciden en las posiciones: [pos:char,...]\n\n")
    
    # Usar Manager para la cola
    with Manager() as manager:
        resultado_queue = manager.Queue()
        procesos = []
        tiempo_inicio = time.perf_counter()
        total_coincidencias = 0
        
        # Lista para la muestra de resultados
        muestra_resultados = []
        
        with open(ruta_archivo, 'r') as archivo, \
             tqdm(total=total_lineas, desc="Procesando archivo") as pbar:
             
            lote_actual = []
            linea_actual = 0
            
            for linea in archivo:
                lote_actual.append(linea)
                linea_actual += 1
                pbar.update(1)
                
                if len(lote_actual) >= tamano_lote or linea_actual == total_lineas:
                    if lote_actual:
                        # Esperar si hay muchos procesos activos
                        while len(procesos) >= num_procesos:
                            for i, p in enumerate(procesos):
                                if not p.is_alive():
                                    procesos.pop(i)
                                    # Procesar resultados
                                    while not resultado_queue.empty():
                                        resultados = resultado_queue.get()
                                        total_coincidencias += len(resultados)
                                        # Guardar en JSON
                                        for res in resultados:
                                            json.dump(res, json_file)
                                            json_file.write('\n')
                                        # Guardar en TXT
                                        for linea1, linea2, coincidencias in resultados:
                                            txt_line = f"Línea {linea1} y {linea2} -> Coinciden en las posiciones: ["
                                            txt_line += ','.join(f"{pos}:{char}" for pos, char in sorted(coincidencias.items()))
                                            txt_line += "]\n"
                                            txt_file.write(txt_line)
                                        # Mantener muestra para mostrar al final
                                        if len(muestra_resultados) < 20:
                                            muestra_resultados.extend(resultados[:20-len(muestra_resultados)])
                                    break
                            else:
                                time.sleep(0.1)
                                continue
                            break
                        
                        # Iniciar nuevo proceso
                        inicio_global = linea_actual - len(lote_actual)
                        p = Process(target=comparar_lineas,
                                  args=(lote_actual, inicio_global, resultado_queue))
                        p.start()
                        procesos.append(p)
                        lote_actual = []
        
        # Esperar a que terminen todos los procesos y procesar resultados restantes
        for p in procesos:
            p.join()
            while not resultado_queue.empty():
                resultados = resultado_queue.get()
                total_coincidencias += len(resultados)
                # Guardar en JSON
                for res in resultados:
                    json.dump(res, json_file)
                    json_file.write('\n')
                # Guardar en TXT
                for linea1, linea2, coincidencias in resultados:
                    txt_line = f"Línea {linea1} y {linea2} -> Coinciden en las posiciones: ["
                    txt_line += ','.join(f"{pos}:{char}" for pos, char in sorted(coincidencias.items()))
                    txt_line += "]\n"
                    txt_file.write(txt_line)
                # Mantener muestra para mostrar al final
                if len(muestra_resultados) < 20:
                    muestra_resultados.extend(resultados[:20-len(muestra_resultados)])
        
        tiempo_total = time.perf_counter() - tiempo_inicio
        json_file.close()
        txt_file.close()
        
        return muestra_resultados, total_coincidencias, tiempo_total

def main():
    ruta_archivo = "genoma.fna"
    num_procesos = min(cpu_count(), 8)  # Limitar a 8 procesos como máximo
    tamano_lote = 5000  # Tamaño de lote
    
    print(f"Iniciando procesamiento con {num_procesos} procesos y lotes de {tamano_lote} líneas")
    
    try:
        muestra_resultados, total_coincidencias, tiempo_total = procesar_archivo_por_lotes(
            ruta_archivo, num_procesos, tamano_lote)
        
        print(f"\nProcesamiento completado en {tiempo_total:.2f} segundos")
        print(f"Total de pares de líneas con coincidencias: {total_coincidencias}")
        
        # Mostrar muestra de resultados
        print("\nMuestra de resultados:")
        for i, (linea1, linea2, coincidencias) in enumerate(muestra_resultados):
            print(f"Línea {linea1} y {linea2} -> Coinciden en las posiciones: [", end="")
            print(','.join(f"{pos}:{char}" for pos, char in sorted(coincidencias.items())), end="")
            print("]")
        
        print(f"\nResultados completos guardados en:")
        print("- 'coincidencias_consecutivas.json' (formato estructurado)")
        print("- 'coincidencias_consecutivas.txt' (formato legible)")
    
    except Exception as e:
        print(f"\nError durante el procesamiento: {str(e)}")
        # Cerrar archivos si están abiertos
        if 'json_file' in locals() and not json_file.closed:
            json_file.close()
        if 'txt_file' in locals() and not txt_file.closed:
            txt_file.close()

if __name__ == '__main__':
    main()