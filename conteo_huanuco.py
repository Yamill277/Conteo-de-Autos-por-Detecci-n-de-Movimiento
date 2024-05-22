import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Cargar el clasificador preentrenado de Haar cascades para detectar cuerpos completos
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Configuración de video
cap = cv2.VideoCapture('autos_huanuco.MP4')

# Definir la región de interés (ROI) para un carril específico
roi_x, roi_y, roi_w, roi_h = 350, 260, 220, 220  

# Línea de conteo (línea horizontal en medio del cuadrado)
line_y = roi_y + roi_h // 2
line_x1, line_x2 = roi_x, roi_x + roi_w

# Variables para contar autos
car_count = 0
cars_detected = set()
recent_cars = []  # Lista temporal para recordar vehículos recientemente detectados
persistence_time = 5  # Tiempo en segundos para recordar vehículos

# Crear la figura y el eje para mostrar el video
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((roi_h, roi_w, 3), dtype=np.uint8))  # Crear una imagen negra inicial

def process_frame(frame, frame_number):
    global car_count, cars_detected, recent_cars

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Definir la región de interés (ROI)
    roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Detectar cuerpos completos en la región de interés (ROI)
    bodies = body_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30), maxSize=(100, 100))

    # Actualizar lista de vehículos recientemente detectados
    recent_cars = [car for car in recent_cars if frame_number - car[0] < persistence_time * 30]

    # Contar autos que cruzan la línea de conteo y están dentro del ROI
    current_cars_detected = set()
    for (x, y, w, h) in bodies:
        # Calcular las coordenadas absolutas del rectángulo en el cuadro completo
        abs_x = roi_x + x
        abs_y = roi_y + y
        abs_xw = abs_x + w
        abs_yh = abs_y + h

        # Dibujar rectángulo alrededor de los cuerpos detectados
        cv2.rectangle(frame, (abs_x, abs_y), (abs_xw, abs_yh), (255, 0, 0), 2)

        # Verificar si el centro del objeto cruza la línea de conteo
        if abs_y < line_y < abs_yh:
            # Verificar si el auto ya fue detectado antes
            already_detected = False
            for (_, (det_x, det_y, det_w, det_h)) in recent_cars:
                if (abs_x >= det_x and abs_xw <= det_x + det_w) and (abs_y >= det_y and abs_yh <= det_y + det_h):
                    already_detected = True
                    break
            
            if not already_detected:
                car_count += 1
                cars_detected.add((abs_x, abs_y, w, h))
                recent_cars.append((frame_number, (abs_x, abs_y, w, h)))
                print(f'Auto detectado! Total: {car_count}')

    # Dibujar cuadrado delimitador en la ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    # Dibujar línea de conteo
    cv2.line(frame, (line_x1, line_y), (line_x2, line_y), (0, 0, 255), 2)

    # Mostrar el conteo de autos detectados
    cv2.putText(frame, f'Autos detectados: {car_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def update(frame_number):
    ret, frame = cap.read()
    if not ret:
        plt.close()  # Cerrar la figura si no hay más frames
        return np.zeros((roi_h, roi_w, 3), dtype=np.uint8)  # Frame negro si no hay frame

    processed_frame = process_frame(frame, frame_number)
    im.set_data(processed_frame)

# Crear la animación
ani = FuncAnimation(fig, update, interval=30)

# Mostrar la figura
plt.title("Conteo de Autos por Detección de Movimiento")
plt.axis('off')  # No mostrar ejes
plt.show()

# Liberar la captura
cap.release()
