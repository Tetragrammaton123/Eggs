import streamlit as st
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
model_path = "best.pt"
model = YOLO(model_path)
def predict(image):
    results = model(image)
    return results[0]
st.title("Классификация яиц: белые и коричневые")
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    st.write("Анализ изображения...")
    result = predict(image_cv)
    im_array = result.plot()
    st.image(im_array, caption="Результаты модели", use_column_width=True)
    egg_counts = {"white egg": 0, "brown egg": 0}
    for box in result.boxes:
        cls = int(box.cls)  # Индекс класса
        class_name = model.names[cls]  # Название класса
        egg_counts[class_name] += 1
    st.write("Распознано:")
    st.write(f"Белые яйца: {egg_counts['white egg']}")
    st.write(f"Коричневые яйца: {egg_counts['brown egg']}")
