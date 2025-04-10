from multiprocessing import Process, Queue, cpu_count, Manager
import time
from tqdm import tqdm
import os

def procesar_lote(lineas, inicio_linea_global, resultado_queue):
    """
    Procesa un lote de líneas y encuentra los caracteres más pequeños
    """
    resultados_lote = []
    numero_linea = inicio_linea_global
    
    for linea in lineas:
        linea = linea.strip()
        numero_linea += 1
        
        if not linea or linea.startswith('>'):
            continue
            
        caracter = min(linea)
        pos = linea.index(caracter) + 1
        resultados_lote.append((numero_linea, pos, caracter))
    
    resultado_queue.put(resultados_lote)

def procesar_archivo_por_lotes(ruta_archivo, num_procesos, tamano_lote=10000):
    """
    Procesa el archivo por lotes usando múltiples procesos
    """
    # Contar líneas totales
    with open(ruta_archivo, 'r') as f:
        total_lineas = sum(1 for _ in f)
    print(f"Total de líneas a procesar: {total_lineas}")
    
    # Usar Manager para la cola (mejor para multiprocesamiento)
    with Manager() as manager:
        resultado_queue = manager.Queue()
        procesos = []
        resultados = []
        tiempo_inicio = time.perf_counter()
        
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
                        # Si hay muchos procesos, esperar a que alguno termine
                        while len(procesos) >= num_procesos:
                            for i, p in enumerate(procesos):
                                if not p.is_alive():
                                    procesos.pop(i)
                                    # Recolectar resultados
                                    while not resultado_queue.empty():
                                        resultados.extend(resultado_queue.get())
                                    break
                            else:
                                time.sleep(0.1)
                                continue
                            break
                        
                        # Iniciar nuevo proceso
                        inicio_global = linea_actual - len(lote_actual)
                        p = Process(target=procesar_lote,
                                  args=(lote_actual, inicio_global, resultado_queue))
                        p.start()
                        procesos.append(p)
                        lote_actual = []
        
        # Esperar a que terminen todos los procesos
        for p in procesos:
            p.join()
            while not resultado_queue.empty():
                resultados.extend(resultado_queue.get())
        
        tiempo_total = time.perf_counter() - tiempo_inicio
        return resultados, tiempo_total

def main():
    ruta_archivo = "genoma.fna"
    num_procesos = min(cpu_count(), 8)  # Limitar a 8 procesos como máximo
    tamano_lote = 5000  # Tamaño de lote más pequeño
    
    print(f"Iniciando procesamiento con {num_procesos} procesos y lotes de {tamano_lote} líneas")
    
    try:
        resultados, tiempo_total = procesar_archivo_por_lotes(ruta_archivo, num_procesos, tamano_lote)
        
        print(f"\nProcesamiento completado en {tiempo_total:.2f} segundos")
        print(f"Total de líneas de secuencia procesadas: {len(resultados)}")
        
        # Muestra de resultados
        muestra_resultados = min(20, len(resultados))
        print("\nMuestra de resultados:")
        for i, (linea, pos, char) in enumerate(resultados[:muestra_resultados]):
            print(f"Línea {linea}: Posición {pos} → '{char}'")
        
        # Guardar resultados
        with open("resultados_por_lotes.txt", "w") as f:
            f.write("Línea\tPosición\tCarácter\n")
            for linea, pos, char in sorted(resultados):
                f.write(f"{linea}\t{pos}\t{char}\n")
        
        print(f"\nResultados completos guardados en 'resultados_por_lotes.txt'")
    
    except Exception as e:
        print(f"\nError durante el procesamiento: {str(e)}")

if __name__ == '__main__':
    main()