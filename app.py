import os
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
image_folder = 'Яйца'
model_path = "best.pt"
model = YOLO(model_path)
def predict(image):
    results = model(image)
    return results[0]
st.title("Классификация яиц: белые и коричневые")
image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
selected_image = st.selectbox("Выберите изображение:", image_files)
image_path = os.path.join(image_folder, selected_image)
image = Image.open(image_path)
st.image(image, caption=f"Выбранное изображение: {selected_image}", use_container_width=True)
image_cv = np.array(image)
image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
st.write("Анализ изображения...")
result = predict(image_cv)
im_array = result.plot()
st.image(im_array, caption="Результаты модели", use_container_width=True)
egg_counts = {"white egg": 0, "brown egg": 0}
for box in result.boxes:
    cls = int(box.cls)
    class_name = model.names[cls]
    egg_counts[class_name] += 1
st.write("Распознано:")
st.write(f"Белые яйца: {egg_counts['white egg']}")
st.write(f"Коричневые яйца: {egg_counts['brown egg']}")
