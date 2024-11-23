import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
 
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) 
 
train_generator = datagen.flow_from_directory( 
    './input/eyes-open-or-closed/dataset/train', 
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='binary', 
    subset='training' 
) 
 
validation_generator = datagen.flow_from_directory( 
    './input/eyes-open-or-closed/dataset/test', 
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='binary', 
    subset='validation' 
)

# model building

model = tf.keras.models.Sequential([ 
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)), 
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), 
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), 
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid') 
]) 
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# training 

model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Real-Time Detection: 

import cv2 
import numpy as np 
import time 
 
cap = cv2.VideoCapture(0) 
 
blink_count = 0 
start_time = time.time() 
sleepiness = 'Not Sleepy' 
 
while True: 
    ret, frame = cap.read() 
    if not ret: 
        break 
     
    resized_frame = cv2.resize(frame, (64, 64)) 
    normalized_frame = resized_frame / 255.0 
    reshaped_frame = np.reshape(normalized_frame, (1, 64, 64, 3)) 
     
    prediction = model.predict(reshaped_frame) 
    eye_state = 'Open' if prediction > 0.5 else 'Closed' 
     
 
    
    if eye_state == 'Closed': 
        blink_count += 1 
         
    elapsed_time = time.time() - start_time 
    remaining_time = max(0, 10 - int(elapsed_time)) 
     
    if elapsed_time > 10: 
        sleepiness = 'Sleepy' if blink_count > 10 else 'Not Sleepy' 
        blink_count = 0 
        start_time = time.time() 
     
    cv2.putText(frame, f'Eye State: {eye_state}', (50, 50), 
cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
    cv2.putText(frame, f'Sleepiness: {sleepiness}', (50, 100), 
cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if sleepiness == 'Not Sleepy' else (0, 0, 
255), 2, cv2.LINE_AA) 
    cv2.putText(frame, f'Time Remaining: {remaining_time}', (50, 150), 
cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 
    cv2.imshow('Eye State Detection', frame) 
     
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
 
cap.release() 
cv2.destroyAllWindows()

