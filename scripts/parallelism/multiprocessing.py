from Bio import SeqIO
from tqdm import tqdm

import os
from tqdm import tqdm

def procesar_grande_con_progreso(ruta_archivo):
    total_lineas = sum(1 for _ in open(ruta_archivo, 'r'))
    
    print(f"Procesando {total_lineas} líneas...")
    
    with open(ruta_archivo, 'r') as archivo, tqdm(total=total_lineas) as pbar:
        for linea in archivo:
            linea = linea.strip()
            if linea.startswith('>'):
                # Procesamiento de encabezado
                pass
            else:
                # Procesamiento de secuencia
                pass
            pbar.update(1)


def contar_registros_fna(ruta_archivo):
    conteo = 0
    for registro in SeqIO.parse(ruta_archivo, "fasta"):
        conteo += 1
    return conteo

def leer_fna_con_biopython(ruta_archivo):
    secuencias = {}
    for registro in SeqIO.parse(ruta_archivo, "fasta"):
        secuencias[registro.id] = str(registro.seq)
    return secuencias

def main():
  archivo_fna = "genoma.fna"
  total_registros = contar_registros_fna(archivo_fna)
  print(f"Total de registros en el archivo: {total_registros}")

  # secuencias = leer_fna_con_biopython(archivo_fna)

  # for iden, seq in secuencias.items():
  #     print(f"ID: {iden}")
  #     print(f"Secuencia (primeros 50 caracteres): {seq[:50]}...")
  #     print(f"Longitud: {len(seq)}")
  #     print("-" * 50)

if __name__ == '__main__':
  main()