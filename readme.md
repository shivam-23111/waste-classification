# ♻️ Waste Classification using MobileNetV2

A deep learning model to classify waste as **Biodegradable** or **Non-Biodegradable** using Transfer Learning (MobileNetV2).

## 📁 Dataset Structure


DATASET/
├── TRAIN/
│   ├── O/  (Organic/Biodegradable images)
│   └── R/  (Recyclable/Non-Biodegradable images)
└── TEST/
    ├── O/
    └── R/



## 🚀 How to Use
1. Prepare dataset in above format
2. Train the model (MobileNetV2 base frozen)
3. Save model as `waste_classification_model.h5`

## 🔍 Prediction
```python
def predict_waste(image_path):
    model = tf.keras.models.load_model('waste_classification_model.h5')
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = np.expand_dims(img, axis=0)
    return 'Biodegradable' if model.predict(img)[0][0] < 0.5 else 'Non-Biodegradable'

