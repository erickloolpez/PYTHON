import cv2
import mediapipe as mp


"""
Captura video desde la webcam, procesa cada frame con MediaPipe Face Mesh (con refine_landmarks activado)
y dibuja:
    - Los puntos de referencia de los ojos (en verde)
    - Los 4 landmarks usados para calcular el centro de la iris (en rojo)
    - El centro calculado de la iris (en azul)
"""
# Inicializar Face Mesh de MediaPipe con refinamiento de landmarks (para iris)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5, 
    refine_landmarks=True  # Activa la detección refinada para la iris
)

# Inicializar la captura de video desde la webcam (índice 0)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame de BGR a RGB (formato requerido por MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(frame_rgb)

    if resultados.multi_face_landmarks:
        for face_landmarks in resultados.multi_face_landmarks:
            # Obtener dimensiones del frame para convertir coordenadas normalizadas a píxeles
            h, w, _ = frame.shape

            # ----------------------------------------------------------------
            # Detección de puntos de referencia de los ojos (índices 33 y 263)
            # ----------------------------------------------------------------
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))
            
            # Dibujar los puntos de los ojos en verde
            cv2.circle(frame, left_eye_coords, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_coords, 3, (0, 255, 0), -1)

            # ----------------------------------------------------------------
            # Detección de los 4 landmarks para el cálculo del centro de la iris
            # ----------------------------------------------------------------
            # Para el ojo izquierdo, se utilizarán los landmarks 468, 469, 470 y 471
            left_iris_points = []
            for i in range(468, 468 + 4):
                pt = face_landmarks.landmark[i]
                x, y = int(pt.x * w), int(pt.y * h)
                left_iris_points.append((x, y))
                # Dibujar cada punto de la iris en rojo
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            # Calcular el centro del iris izquierdo promediando las coordenadas de los 4 puntos
            left_iris_center = (
                int(sum([p[0] for p in left_iris_points]) / len(left_iris_points)),
                int(sum([p[1] for p in left_iris_points]) / len(left_iris_points))
            )

            # Para el ojo derecho, se utilizarán los landmarks 473, 474, 475 y 476
            right_iris_points = []
            for i in range(473, 473 + 4):
                pt = face_landmarks.landmark[i]
                x, y = int(pt.x * w), int(pt.y * h)
                right_iris_points.append((x, y))
                # Dibujar cada punto de la iris en rojo
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            # Calcular el centro del iris derecho
            right_iris_center = (
                int(sum([p[0] for p in right_iris_points]) / len(right_iris_points)),
                int(sum([p[1] for p in right_iris_points]) / len(right_iris_points))
            )

            # Dibujar el centro de la iris (pupila) en azul para cada ojo
            cv2.circle(frame, left_iris_center, 3, (255, 0, 0), -1)
            cv2.circle(frame, right_iris_center, 3, (255, 0, 0), -1)

    # Mostrar el frame resultante con las marcas
    cv2.imshow("Face Mesh - Ojos y Pupilas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
