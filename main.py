import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# สร้างโมเดล CNN
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')  # 4 คลาส: dark, light, mid-dark, mid-light
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_data(train_dir, val_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),  # ปรับขนาดรูปภาพให้ตรงกับโมเดล
        batch_size=8,
        class_mode='categorical',
        shuffle=True  # สุ่มข้อมูลสำหรับการฝึก
    )
    
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),  # ปรับขนาดรูปภาพให้ตรงกับโมเดล
        batch_size=8,
        class_mode='categorical'
    )
    
    return train_generator, val_generator

def train_model(model, train_generator, val_generator, epochs=10):  # เพิ่มจำนวน epoch สำหรับการฝึก
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )
    return history

def predict_skin_tone(model, image_path):
    image = cv2.imread(image_path)
    
    # ตรวจสอบว่าภาพถูกโหลดหรือไม่
    if image is None:
        raise ValueError(f"ไม่สามารถโหลดภาพจาก {image_path} ได้ กรุณาตรวจสอบไฟล์ภาพ")

    # ปรับขนาดภาพเป็น 224x224
    image = cv2.resize(image, (224, 224))
    
    # ขยายมิติของภาพและ normalize
    image = np.expand_dims(image, axis=0) / 255.0

    # ทำนายเฉดสีผิว
    prediction = model.predict(image)
    
    # คำนวณคลาสที่มีค่าความน่าจะเป็นสูงสุด
    skin_tone_class = np.argmax(prediction)

    # คลาสของสีผิวที่ทำนายได้
    classes = ['dark', 'light', 'mid-dark', 'mid-light']
    return classes[skin_tone_class]

def load_color_palette_images(season_palette_dir, skin_tone):
    palette_path = os.path.join(season_palette_dir, f"{skin_tone}_skin_colors")
    palette_images = []
    
    if not os.path.exists(palette_path):
        raise ValueError(f"ไม่พบโฟลเดอร์สำหรับสีผิว: {palette_path}")
    
    for filename in os.listdir(palette_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(palette_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # แปลงสีจาก BGR เป็น RGB
                palette_images.append(image)
    return palette_images

def main():
    train_dir = "skin_tone_dataset/train"
    val_dir = "skin_tone_dataset/val"
    
    input_shape = (224, 224, 3)  # ขนาดภาพที่ใช้ในโมเดล
    model = create_cnn_model(input_shape)
    
    train_generator, val_generator = prepare_data(train_dir, val_dir)
    
    print("เริ่มเทรนโมเดล...")
    history = train_model(model, train_generator, val_generator, epochs=10)  # ใช้ 10 epochs สำหรับการฝึก

    # บันทึกโมเดล
    model_path = os.path.join(os.getcwd(), "skin_tone_cnn_model.h5")
    model.save(model_path)
    print(f"โมเดลถูกบันทึกไว้ที่ {model_path}")
    
    # ทำนายสีผิวจากภาพของผู้ใช้
    user_image_path = input("กรุณาใส่เส้นทางรูปภาพสีผิวของคุณ: ")
    predicted_tone = predict_skin_tone(model, user_image_path)
    print(f"สีผิวที่คาดว่า: {predicted_tone}")
    
    season_palette_dir = "color_palette_dataset"
    suitable_colors_images = load_color_palette_images(season_palette_dir, predicted_tone)
    
    # แสดงเฉดสีที่เหมาะสม
    print("เฉดสีที่เหมาะสม:")
    for idx, color_image in enumerate(suitable_colors_images):
        plt.subplot(1, len(suitable_colors_images), idx + 1)
        plt.imshow(color_image)
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
