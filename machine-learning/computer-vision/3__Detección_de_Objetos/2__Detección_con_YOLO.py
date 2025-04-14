import cv2
import time
from ultralytics import YOLO


# Cargar el modelo YOLO v11
model = YOLO("yolo11n.pt")

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

# Configurar la ventana para que se muestre en pantalla completa
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Medir el tiempo de procesamiento para calcular la latencia
    start_time = time.time()
    
    # Realizar la detección en el frame
    results = model(
        frame, 
        #conf=0.4,
        #classes=[0]
        )
    
    # Calcular la latencia en milisegundos
    latency = (time.time() - start_time) * 1000

    # Acceder al primer resultado (YOLOv11 devuelve una lista de resultados)
    boxes_obj = results[0].boxes

    # Si se detectaron objetos, extraer información y dibujar sobre el frame
    if boxes_obj is not None and len(boxes_obj) > 0:
        # Extraer las cajas delimitadoras, las puntuaciones de confianza y los índices de clase
        bboxes = boxes_obj.xyxy.cpu().numpy()   # Formato: [x1, y1, x2, y2] para cada detección
        confs = boxes_obj.conf.cpu().numpy()      # Puntajes de confianza
        classes = boxes_obj.cls.cpu().numpy()     # Índices de clase
        
        # Iterar sobre cada detección y dibujarla en el frame
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Obtener el nombre de la clase; si el modelo no tiene 'names', usar el índice como string
            class_name = model.names[int(classes[i])] if hasattr(model, 'names') else str(int(classes[i]))
            
            # Crear la etiqueta que incluye el nombre de la clase y el valor de confianza
            label = f'{class_name} {confs[i]:.2f}'
            
            # Dibujar el rectángulo en el frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Dibujar la etiqueta justo arriba de la caja
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Mostrar la latencia en el frame para conocer el rendimiento del sistema
    cv2.putText(frame, f'Latency: {latency:.1f}ms', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Mostrar el frame procesado en tiempo real
    cv2.imshow("YOLOv11 - Deteccion en Tiempo Real", frame)
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
