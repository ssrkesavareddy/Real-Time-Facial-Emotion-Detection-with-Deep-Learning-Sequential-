#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + r"\haarcascade_frontalface_default.xml")


# In[3]:


import time


model = load_model(r"C:\Users\ssrke\Downloads\emotionsdatasets\model\model.h5")


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


cap = cv2.VideoCapture(0)

start_time = time.time()
duration = 90  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    elapsed_time = time.time() - start_time

 
    time_remaining = max(duration - elapsed_time, 0)
    cv2.putText(frame, f"Time Remaining: {time_remaining:.1f} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    if elapsed_time >= duration:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

   
        predictions = model.predict(roi)[0]
        emotion_probabilities = zip(emotion_labels, predictions)
        emotion_probabilities = sorted(emotion_probabilities, key=lambda x: x[1], reverse=True)

        label = f"{emotion_probabilities[0][0]}: {emotion_probabilities[0][1]:.2f}"
        label_position = (x, y)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('Emotion Detector', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time >= duration:
        break


cap.release()
cv2.destroyAllWindows()


# In[ ]:




