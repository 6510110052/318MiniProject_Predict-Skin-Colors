import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template

# สร้างแอป Flask
app = Flask(__name__)

# โหลดโมเดลที่เทรนไว้แล้ว
model = load_model("skin_tone_cnn_model.h5")

# ฟังก์ชันทำนายเฉดสีผิว
def predict_skin_tone(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"ไม่สามารถโหลดภาพจาก {image_path} ได้ กรุณาตรวจสอบไฟล์ภาพอีกครั้ง")

    image = cv2.resize(image, (224, 224))
    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    skin_tone_class = np.argmax(prediction)
    classes = ['dark', 'light', 'mid-dark', 'mid-light']
    return classes[skin_tone_class]

# ฟังก์ชันแสดงเฉดสีที่เหมาะสม
def load_color_palette(season_palette_dir, predicted_tone):
    suitable_colors = []
    color_folder = os.path.join(season_palette_dir, f"{predicted_tone}_skin_colors")

    # ตรวจสอบว่าโฟลเดอร์มีอยู่จริง
    if not os.path.exists(color_folder):
        raise ValueError(f"โฟลเดอร์สำหรับโทนสี {predicted_tone} ไม่พบ")

    for filename in os.listdir(color_folder):
        if filename.endswith('.png'):
            color_name = filename[:-4]  # สมมติชื่อสีเป็นชื่อไฟล์โดยตัด '.png' ออก
            color_hex = color_name.strip('#')  # ลบเครื่องหมาย # ออกหากมี
            suitable_colors.append((color_hex, filename))  # เพิ่มชื่อสีและเส้นทางไฟล์

    return suitable_colors

# Route สำหรับหน้าเว็บ
@app.route('/')
def index():
    return render_template('index.html')

# เส้นทางในการเก็บรูปภาพ
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# API สำหรับทำนายเฉดสีผิวและเฉดสีที่เหมาะสม
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'กรุณาอัปโหลดไฟล์รูปภาพ'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'ไม่มีไฟล์ถูกเลือก'})

    # บันทึกรูปภาพที่อัปโหลด
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # ทำนายเฉดสีผิว
    try:
        predicted_tone = predict_skin_tone(file_path)
    except ValueError as e:
        return jsonify({'error': str(e)})

    # โหลดเฉดสีที่เหมาะสม
    season_palette_dir = "color_palette_dataset"
    suitable_colors = load_color_palette(season_palette_dir, predicted_tone)
    
    # ลบรูปภาพที่อัปโหลดออกหลังการประมวลผลเสร็จสิ้น
    os.remove(file_path)

    return jsonify({
        'predicted_tone': predicted_tone,
        'suitable_colors': suitable_colors
    })

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)
