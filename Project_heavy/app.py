import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

# โหลดโมเดลที่ฝึกแล้ว
model = load_model('C:/Users/AsusT/Python/Project_heavy-metals/model/46ClassHSVA3C_fold4_final.keras')

# กำหนด label ของคลาสต่างๆ
class_labels = ['Arsenic0.007mg', 'Arsenic0.015mg', 'Arsenic0.03mg', 'Arsenic0.06mg', 'Arsenic0.12mg', 
                'Arsenic0.25mg', 'Arsenic0.50mg', 'Arsenic1.00mg', 'Arsenic2.00mg', 'Arsenic4.00mg',
                'Cadmium0.00022mg', 'Cadmium0.0004mg', 'Cadmium0.00077mg', 'Cadmium0.001mg', 
                'Cadmium0.003mg', 'Cadmium0.006mg', 'Cadmium0.012mg', 'Cadmium0.025mg', 
                'Cadmium0.05mg', 'Copper0.35mg', 'Copper0.75mg', 'Copper1mg', 'Copper1.5mg', 
                'Copper3mg', 'Copper6mg', 'Iron1mg', 'Iron0.1mg', 'Iron0.3mg', 'Iron1.00mg', 
                'Iron2.00mg', 'Iron3mg', 'Iron4mg', 'Iron6mg', 'Iron8mg', 'Lead0.006mg', 
                'Lead0.015mg', 'Lead0.025mg', 'Lead0.05mg', 'Lead0.1mg', 'Manganese0.006mg', 
                'Manganese0.01mg', 'Manganese0.02mg', 'Manganese0.04mg', 'Manganese0.05mg', 
                'Manganese0.08mg', 'Manganese0.1mg']

# ฟังก์ชันสำหรับทำนายผลจากภาพ
def predict_image(image):
    # ปรับขนาดและเตรียมข้อมูลภาพ
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    # ทำนายผล
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_labels[predicted_class]

# ส่วน UI
st.title("ทำนายผลจากรูปภาพ")
uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(100, 100))
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)
    
    # ทำนายผล
    prediction = predict_image(image)
    st.write(f"ผลลัพธ์การทำนาย: {prediction}")
