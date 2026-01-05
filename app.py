"""
Fruit Recognition Web Application
Upload an image and get real-time fruit classification
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json

# Support for HEIC images
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'heic', 'HEIC'}

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global variables for models
cnn_model = None
transfer_model = None
class_names = None

# Load class names from latest model info
def load_class_names():
    """Load class names from model info JSON"""
    global class_names
    try:
        # Find latest model info file
        model_info_files = [f for f in os.listdir('models') if f.endswith('_model_info_.json')]
        if model_info_files:
            latest = sorted(model_info_files)[-1]
            with open(os.path.join('models', latest), 'r') as f:
                info = json.load(f)
                class_names = info.get('class_names', [])
                return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
    
    # Default class names if not found
    return ['Apple', 'Avocado', 'Banana', 'Lemon', 'Mango', 'Orange', 'Papaya', 'Pineapple', 'Tomato', 'Watermelon']


def load_models():
    """Load the trained models"""
    global cnn_model, transfer_model, class_names
    
    class_names = load_class_names()
    
    # Load CNN model
    cnn_files = [f for f in os.listdir('models') if f.startswith('cnn_model_') and f.endswith('_best.h5')]
    if cnn_files:
        latest_cnn = sorted(cnn_files)[-1]
        cnn_path = os.path.join('models', latest_cnn)
        try:
            cnn_model = keras.models.load_model(cnn_path)
            print(f"✅ Loaded CNN model: {latest_cnn}")
        except Exception as e:
            print(f"❌ Error loading CNN model: {e}")
    
    # Load Transfer Learning model
    transfer_files = [f for f in os.listdir('models') if f.startswith('transfer_MobileNetV2_model_') and f.endswith('_best.h5')]
    if transfer_files:
        latest_transfer = sorted(transfer_files)[-1]
        transfer_path = os.path.join('models', latest_transfer)
        try:
            transfer_model = keras.models.load_model(transfer_path)
            print(f"✅ Loaded Transfer Learning model: {latest_transfer}")
        except Exception as e:
            print(f"❌ Error loading Transfer Learning model: {e}")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image_path, img_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Open image (supports HEIC)
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(img_size)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_fruit(model, image_array, model_type='cnn'):
    """Make prediction using the model"""
    try:
        predictions = model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'class': class_names[idx] if idx < len(class_names) else f'Class {idx}',
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        return {
            'predicted_class': class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f'Class {predicted_class_idx}',
            'confidence': confidence,
            'top_3': top_3_predictions,
            'all_predictions': {class_names[i] if i < len(class_names) else f'Class {i}': float(predictions[0][i]) 
                               for i in range(len(predictions[0]))}
        }
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         cnn_available=cnn_model is not None,
                         transfer_available=transfer_model is not None,
                         class_names=class_names)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    model_type = request.form.get('model_type', 'transfer')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        image_array = preprocess_image(filepath)
        if image_array is None:
            return jsonify({'error': 'Error preprocessing image'}), 400
        
        # Select model
        model = transfer_model if model_type == 'transfer' else cnn_model
        
        if model is None:
            return jsonify({'error': f'{model_type} model not available'}), 404
        
        # Make prediction
        result = predict_fruit(model, image_array, model_type)
        
        if result is None:
            return jsonify({'error': 'Error making prediction'}), 500
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'prediction': result,
            'image_filename': filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/models_status')
def models_status():
    """Get status of available models"""
    return jsonify({
        'cnn_available': cnn_model is not None,
        'transfer_available': transfer_model is not None,
        'class_names': class_names
    })


if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("\n🚀 Starting Fruit Recognition Web App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("📸 Upload a fruit image to get predictions!")
    print("⚠️  If port 5000 is busy, try: http://127.0.0.1:5000")
    print("⚠️  Press Ctrl+C to stop the server\n")
    try:
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ Port 5000 is already in use!")
            print("💡 Try these solutions:")
            print("   1. Kill existing process: lsof -ti:5000 | xargs kill -9")
            print("   2. Use different port: python3 -c \"from app import app; app.run(port=5001)\"")
        else:
            print(f"\n❌ Error starting server: {e}")

