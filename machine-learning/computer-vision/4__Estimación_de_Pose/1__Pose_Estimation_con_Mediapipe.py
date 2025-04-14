import cv2
import mediapipe as mp


# Inicializar Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose( 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    static_image_mode=False,
    )

# Inicializar dibujo de landmarks
mp_drawing = mp.solutions.drawing_utils

# Capturar video en tiempo real desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir frame a RGB (requerido por Mediapipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar imagen para obtener landmarks
    results = pose.process(frame_rgb)
    
    # Dibujar landmarks en la imagen si se detectan
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Mostrar el frame con los landmarks
    cv2.imshow('Pose Estimation', frame)
    
    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
