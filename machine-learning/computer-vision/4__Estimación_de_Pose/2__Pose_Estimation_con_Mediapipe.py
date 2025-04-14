import cv2
import mediapipe as mp

# Inicializar Face Mesh de Mediapipe (Nos centramos en la cara)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capturar video de la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(frame_rgb)

    if resultados.multi_face_landmarks:
        for face_landmarks in resultados.multi_face_landmarks:
            # Obtener puntos clave de los ojos
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            
            # Pasarla a coordenadas absolutas
            h, w, _ = frame.shape
            left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))

            # Dibujar los ojos en la imagen
            cv2.circle(frame, left_eye_coords, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_coords, 3, (0, 255, 0), -1)

    cv2.imshow("Face Mesh - Ojos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
