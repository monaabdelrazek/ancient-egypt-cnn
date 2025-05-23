from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import os
from PIL import Image
import tensorflow as tf


app = Flask(__name__)

# تحميل الموديل
MODEL_PATH = 'brave_pharos_detection_model256.keras'
model = load_model(MODEL_PATH, compile=False)

CLASS_NAMES = ['King Akhenaten',
               'King AmenhotepIII',
               ' Bent pyramid of senefru',
               'Colossoi of Memnon',
               'Goddess Isis',
               'Queen Hatshepsut',
               'Khafre Pyramid',
               'King Thutmose III',
               'King Tutankhamun',
               'Queen Nefertiti',
               'Pyramid_of_Djoser',
               'King Ramesses II',
               'Ramessum( Memorial temple of Ramesses II )',
               'King Zoser',
               'Tutankhamun with Ankhesenamun',
               'Temple_of_Hatshepsut',
               'Temple_of_Isis_in_Philae',
               'Temple_of_Kom_Ombo',
               'The Great Temple of Ramesses II',
               'menkaure pyramid',
               'sphinx']

# === باقي كود API ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        img = Image.open(file.stream).convert('RGB').resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model.predict(img_array)
        return jsonify({
            'class': CLASS_NAMES[np.argmax(pred)],
            'confidence': float(np.max(pred))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)