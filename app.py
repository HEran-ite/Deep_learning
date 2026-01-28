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
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'heic', 'HEIC'}


# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

from src.data_preprocessing import preprocess_image as unify_preprocess


cnn_model = None
class_names = None

# Global variable to store model image size
model_img_size = (192, 192)


# Load class names from latest model info
def load_class_names():
    """Load class names from model info JSON"""
    global class_names, model_img_size
    try:
        # Find latest model info file (cnn_from_scratch_info_*.json)
        model_info_files = [f for f in os.listdir('models') if f.startswith('cnn_from_scratch_info_') and f.endswith('.json')]
        if model_info_files:
            latest = sorted(model_info_files)[-1]
            with open(os.path.join('models', latest), 'r') as f:
                info = json.load(f)
                class_names = info.get('class_names', [])
                # Get image size from model info
                img_size = info.get('img_size', [192, 192])
                if isinstance(img_size, list) and len(img_size) == 2:
                    model_img_size = tuple(img_size)
                print(f"Loaded model info: {len(class_names)} classes, image size: {model_img_size}")
                return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
    
    # Default class names if not found
    return ['Apple', 'Avocado', 'Banana', 'Lemon', 'Mango', 'Orange', 'Papaya', 'Pineapple', 'Tomato', 'Watermelon']


def load_models():
    """Load the trained CNN model"""
    global cnn_model, class_names, model_img_size
    
    class_names = load_class_names()
    
    # Load CNN from scratch model (best model)
    cnn_files = [f for f in os.listdir('models') if f.startswith('cnn_from_scratch_') and f.endswith('_best.h5')]
    if cnn_files:
        latest_cnn = sorted(cnn_files)[-1]
        cnn_path = os.path.join('models', latest_cnn)
        try:
            cnn_model = keras.models.load_model(cnn_path)
            print(f"Loaded CNN from scratch model: {latest_cnn}")
            print(f"   Model image size: {model_img_size}")
        except Exception as e:
            print(f"Error loading CNN model: {e}")
    else:
        print("  No CNN from scratch model found")
    
    # No transfer learning models are used in this version (CNN-only)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image_path, img_size=None):
    """Preprocess image for model prediction using unified logic"""
    global model_img_size
    if img_size is None:
        img_size = model_img_size
    
    # Use the unified preprocessing from src.data_preprocessing
    # Note: rescale=False because the model has a Rescaling(1./255) layer
    img_array = unify_preprocess(image_path, img_size=img_size, rescale=False)
    
    if img_array is not None:
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
    return img_array


def predict_fruit(model, image_array, model_type='cnn'):
    """Make prediction using the model"""
    try:
        # Debug: Check input range
        print(f"Debug: Input image min={image_array.min():.2f}, max={image_array.max():.2f}, mean={image_array.mean():.2f}")
        
        predictions = model.predict(image_array, verbose=0)
        
        # Debug: Print raw predictions
        print(f"Debug: Raw predictions: {predictions[0]}")
        print(f"Debug: Prediction sum: {predictions[0].sum():.4f}")
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        print(f"Debug: Predicted class index: {predicted_class_idx}, Class: {class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else 'Unknown'}, Confidence: {confidence:.4f}")
        
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
        import traceback
        traceback.print_exc()
        return None


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         cnn_available=cnn_model is not None,
                         transfer_available=False,
                         class_names=class_names)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: jpg, jpeg, png, heic'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        image_array = preprocess_image(filepath)
        if image_array is None:
            return jsonify({'error': 'Error preprocessing image'}), 400
        
        # Select model (CNN only)
        if cnn_model is None:
            return jsonify({'error': 'No model available'}), 404
        model = cnn_model
        
        # Make prediction
        result = predict_fruit(model, image_array, model_type='cnn')
        
        if result is None:
            return jsonify({'error': 'Error making prediction'}), 500
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'model_type': 'cnn',
            'prediction': result,
            'image_filename': filename
        })
    except Exception as e:
        # Ensure we always return JSON, not HTML
        import traceback
        error_msg = str(e)
        if app.debug:
            error_msg += '\n' + traceback.format_exc()
        return jsonify({'error': f'Server error: {error_msg}'}), 500


@app.route('/models_status')
def models_status():
    """Get status of available models"""
    return jsonify({
        'cnn_available': cnn_model is not None,
        'transfer_available': False,
        'class_names': class_names
    })


# Error handlers to return JSON instead of HTML
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error: ' + str(error)}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request: ' + str(error)}), 400

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Please upload an image smaller than 32MB.'}), 413

if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("\n Starting Fruit Recognition Web App...")
    print(" Open your browser and go to: http://localhost:5000")
    print(" Upload a fruit image to get predictions!")
    print(" If port 5000 is busy, try: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop the server\n")
    try:
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n Port 5000 is already in use!")
            print(" Try these solutions:")
            print("   1. Kill existing process: lsof -ti:5000 | xargs kill -9")
            print("   2. Use different port: python3 -c \"from app import app; app.run(port=5001)\"")
        else:
            print(f"\n Error starting server: {e}")

