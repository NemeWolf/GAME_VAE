import cv2
import numpy as np

def capture_frames(video_path, frame_interval=30, target_size=(28, 28)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        # ret es un booleano que indica si el frame se ha leído correctamente
        # frame es el frame leído --> dimensiones: (alto, ancho, canales)
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Redimensionar el frame
            frame = cv2.resize(frame, target_size)
            # Convertir a escala de grises
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Normalizar el frame
            frame = frame.astype('float32') / 255.0
            # Añadir una dimensión para el canal
            frame = np.expand_dims(frame, axis=-1)
            frames.append(frame)

        frame_count += 1

    cap.release()
    return np.array(frames)

# Ejemplo de uso
video_path = 'data\IMG\VIDEO_DATASET.mp4'
frames = capture_frames(video_path)
print(f'Número de frames capturados: {frames.shape[0]}')


# PCA, ICA --> Componentes --> Varianza explicada
# Analizar cantidad de componentes y seleccionar numero de variables latentes