import cv2
import mediapipe as mp
import numpy as np


def add_gaussian_to_heatmap(heatmap, center, sigma, amplitude):
    """
    Agrega un parche gaussiano al heatmap en la posición especificada.
    
    Parámetros:
        heatmap (numpy.ndarray): Matriz de activación (h x w) que representa la distribución de mirada.
        center (tuple): Coordenadas (cx, cy) del centro del parche.
        sigma (int): Desviación estándar de la distribución gaussiana.
        amplitude (int): Máximo valor de intensidad a sumar.
    
    Retorna:
        numpy.ndarray: Heatmap actualizado con el nuevo parche gaussiano.
    """
    h, w = heatmap.shape
    y, x = np.indices((h, w))  # Se generan matrices de índices para coordenadas
    cx, cy = center
    
    # Se calcula la distribución gaussiana centrada en (cx, cy)
    gaussian = amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    
    heatmap += gaussian  # Se suma la distribución al heatmap existente
    return heatmap


# Parámetros de configuración

# Factor de decaimiento para el heatmap, que permite que las activaciones antiguas desaparezcan gradualmente
# Valores cercanos a 1 hacen que el rastro de la mirada dure más tiempo
decay_factor = 0.98  

# Tamaño de la dispersión del parche gaussiano aplicado al heatmap
sigma = 15  

# Intensidad máxima de cada parche gaussiano agregado al heatmap
amplitude = 50  

# Inicialización del modelo de Face Mesh de Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de video en tiempo real desde la cámara web
cap = cv2.VideoCapture(0)

# Configurar la ventana para que se muestre en pantalla completa
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ret, frame = cap.read()
h, w, _ = frame.shape  # Se obtienen dimensiones del frame

# Inicialización del heatmap de seguimiento de miradas (matriz con valores inicializados en 0)
heatmap_gaze = np.zeros((h, w), dtype=np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Salir del bucle si no hay más frames disponibles

    # Aplicar decaimiento exponencial al heatmap para reducir intensidad de activaciones antiguas
    heatmap_gaze *= decay_factor

    # Convertir el frame de BGR a RGB para procesarlo con Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(frame_rgb)

    if resultados.multi_face_landmarks:
        for face_landmarks in resultados.multi_face_landmarks:
            # Se extraen los puntos clave de los ojos
            left_eye = face_landmarks.landmark[33]  # Punto correspondiente al ojo izquierdo
            right_eye = face_landmarks.landmark[263]  # Punto correspondiente al ojo derecho

            # Se convierten las coordenadas normalizadas (0-1) a píxeles en la imagen
            left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))

            # Se calcula el punto medio entre ambos ojos como una estimación de la dirección de la mirada
            mid_eye = ((left_eye_coords[0] + right_eye_coords[0]) // 2,
                       (left_eye_coords[1] + right_eye_coords[1]) // 2)

            # Se agrega un parche gaussiano en el heatmap en la ubicación de la mirada
            heatmap_gaze = add_gaussian_to_heatmap(heatmap_gaze, mid_eye, sigma, amplitude)

    # Normalización del heatmap para escalar los valores entre 0 y 255
    heatmap_vis = cv2.normalize(heatmap_gaze, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_vis = np.uint8(heatmap_vis)  # Convertir a tipo de dato uint8
    colored_heatmap = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)  # Aplicar mapa de colores tipo 'JET'

    # Superponer el heatmap coloreado sobre el frame original con transparencia
    overlay = cv2.addWeighted(frame, 0.6, colored_heatmap, 0.4, 0)

    # Mostrar el frame con la visualización del heatmap
    cv2.imshow("Heatmap de Mirada", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Salir si se presiona la tecla 'q'

# Liberar recursos al finalizar
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
