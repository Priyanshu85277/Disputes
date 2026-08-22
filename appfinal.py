from flask import Flask, render_template, request, send_file, jsonify, send_from_directory, session, redirect, url_for
from fpdf import FPDF
from PIL import Image
import os
import json
import torch
from ultralytics import YOLO
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from flask import send_from_directory
import uuid
from datetime import datetime
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v1 as IFD
import piexif
from PIL.ExifTags import TAGS, GPSTAGS

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'deepanshu1234'  # Replace with a secure random key

# Configurations
UPLOAD_FOLDER = 'static/uploads'
PREDICTION_FOLDER = 'static/predictions'
REPORT_FOLDER = 'static/reports'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MongoDB connection
MONGO_URI = os.environ.get(
    'MONGO_URI',
    ''
)

# Only attempt a MongoDB connection when an explicit MONGO_URI is provided.
client = None
users_collection = None
USERS_JSON_FILE = os.path.join(BASE_DIR, 'local_users.json')

def load_local_users():
    if os.path.exists(USERS_JSON_FILE):
        try:
            with open(USERS_JSON_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            logging.exception('Failed to read local users file. Starting with empty user store.')
    return []


def save_local_users(users):
    try:
        with open(USERS_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2)
    except Exception:
        logging.exception('Failed to write local users file.')


def get_user_by_email(email):
    if users_collection is not None:
        return users_collection.find_one({'email': email})
    users = load_local_users()
    return next((u for u in users if u.get('email') == email), None)


def create_local_user(name, email, password):
    users = load_local_users()
    users.append({'name': name, 'email': email, 'password': password})
    save_local_users(users)


if not MONGO_URI:
    logging.info('No MONGO_URI provided; MongoDB features are disabled. Using local authentication fallback.')
else:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client.get_default_database() or client['DeepanshuPro']
        users_collection = db['users']
        logging.info('Connected to MongoDB successfully.')
    except PyMongoError:
        logging.exception('MongoDB connection failed. Authentication, DNS, or network may be unavailable. Falling back to local authentication store.')
        client = None
        users_collection = None

# YOLO models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
preferred_model = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'best2.pt'))
if not os.path.exists(preferred_model):
    logging.warning('Preferred local YOLO model not found at %s', preferred_model)
    for candidate in ['best.pt', 'best2.pt', 'best3.pt', 'best4.pt', 'yolov5s.pt']:
        candidate_path = os.path.join(BASE_DIR, candidate)
        if os.path.exists(candidate_path):
            preferred_model = candidate_path
            logging.warning('Using fallback YOLO model file %s', candidate_path)
            break

if os.path.exists(preferred_model):
    try:
        model = YOLO(preferred_model)
        logging.info('Loaded YOLO model from %s', preferred_model)
    except Exception:
        logging.exception('Failed to load YOLO model from %s', preferred_model)
        model = None
else:
    logging.error('No local YOLO model found at %s; model inference will be disabled.', preferred_model)
    model = None

torch_model_path = os.path.join(BASE_DIR, 'yolov5s.pt')
if os.path.exists(torch_model_path):
    try:
        torch_model = YOLO(torch_model_path)
    except Exception:
        logging.exception('Failed to load report YOLO model from %s. Report generation will be unavailable.', torch_model_path)
        torch_model = None
else:
    logging.error('Local report YOLO model not found at %s; report generation will be disabled.', torch_model_path)
    torch_model = None

# Disable decompression bomb error for large images
Image.MAX_IMAGE_PIXELS = None

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def convert_to_dms(decimal):
    """Convert decimal coordinates to degrees, minutes, and seconds."""
    degrees = int(decimal)
    minutes = int((decimal - degrees) * 60)
    seconds = int(((decimal - degrees) * 60 - minutes) * 60 * 100)
    return [(degrees, 1), (minutes, 1), (seconds, 100)]

def extract_geolocation(image_path):
    try:
        # Open the image
        image = Image.open(image_path)
        exif_data = image._getexif()  # Extract EXIF metadata

        if not exif_data:
            print("No EXIF data found.")
            return None

        # Log EXIF data for debugging
        print("EXIF data:", exif_data)

        # Extract GPSInfo tag (34853 is the GPS tag ID)
        gps_info = exif_data.get(34853)
        if not gps_info:
            print("No GPSInfo found in EXIF data.")
            return None

        # Log GPS data for debugging
        print("GPSInfo:", gps_info)

        # Convert GPS data to decimal format
        def convert_to_decimal(values, ref):
            degrees = values[0][0] / values[0][1]
            minutes = values[1][0] / values[1][1]
            seconds = values[2][0] / values[2][1] if len(values) > 2 else 0
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            if ref in ['S', 'W']:
                decimal = -decimal
            return decimal

        latitude = convert_to_decimal(gps_info[2], gps_info[1])
        longitude = convert_to_decimal(gps_info[4], gps_info[3])

        return latitude, longitude

    except Exception as e:
        print(f"Error extracting geolocation: {e}")
        return None

@app.route('/upload_metadata_image', methods=['POST'])
def upload_metadata_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Extract geolocation metadata
        geolocation = extract_geolocation(file_path)
        if not geolocation:
            return jsonify({
                "error": "No geolocation metadata found in the image. "
                         "Ensure the image contains EXIF geolocation data."
            }), 400

        latitude, longitude = geolocation
        return jsonify({"latitude": latitude, "longitude": longitude})

    except Exception as e:
        logging.exception("An error occurred while processing the metadata image.")
        return jsonify({"error": str(e)}), 500


def add_geolocation(image_path, latitude, longitude):
    try:
        # Open the image
        image = Image.open(image_path)

        # Check if EXIF exists, if not create it
        exif_data = image.info.get("exif", None)
        if exif_data:
            exif_dict = piexif.load(exif_data)
        else:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}

        # Prepare GPS data
        exif_dict["GPS"] = {
            piexif.GPSIFD.GPSLatitudeRef: b'N' if latitude >= 0 else b'S',
            piexif.GPSIFD.GPSLatitude: convert_to_dms(abs(latitude)),
            piexif.GPSIFD.GPSLongitudeRef: b'E' if longitude >= 0 else b'W',
            piexif.GPSIFD.GPSLongitude: convert_to_dms(abs(longitude)),
        }

        # Convert EXIF data back to bytes
        exif_bytes = piexif.dump(exif_dict)


        # Save the image as JPEG if it is not already
        image = image.convert("RGB")  # Convert to RGB if not already (for saving as JPEG)
        image.save(image_path, "JPEG", exif=exif_bytes)
        print(f"Geolocation metadata added to image: {image_path}")

    except Exception as e:
        print(f"Error: {e}")


@app.route('/')
def index():
    return render_template('landing.html')


# User Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    return redirect(url_for('dashboard'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    user_name = session.get('name', 'User')
    return render_template('dashboard.html', user_name=user_name)



# Object Detection and PDF Report Generation
@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    if torch_model is None:
        return jsonify({"error": "Report generation model is unavailable."}), 503
    try:
        image = Image.open(image_path).convert('RGB')
        if max(image.size) > 1024:
            image = image.resize((1024, int(1024 * image.size[1] / image.size[0])))

        results = torch_model(image)
        labels = results.names
        boxes = results.xywh[0].tolist()
        object_counts = {}

        for box in boxes:
            label = labels[int(box[5])]
            object_counts[label] = object_counts.get(label, 0) + 1
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400

    report_path = os.path.join(REPORT_FOLDER, "image_analysis_report.pdf")
    generate_pdf_report(report_path, object_counts, image_path)
    return send_file(report_path, as_attachment=True, mimetype='application/pdf')


def generate_pdf_report(report_path, object_counts, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Object Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Uploaded Image:", ln=True)
    pdf.image(image_path, x=10, y=30, w=190)
    pdf.ln(80)
    pdf.cell(0, 10, txt="Detected Objects:", ln=True)
    for label, count in object_counts.items():
        pdf.cell(0, 10, txt=f"- {label}: {count}", ln=True)
    pdf.output(report_path)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Object detection model is unavailable."}), 503
        if 'file' not in request.files or not request.form.get('latitude') or not request.form.get('longitude'):
            return jsonify({"error": "Please provide an image and geolocation data"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])


        # Generate a unique filename for the upload
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(upload_path)
        logging.info(f"Uploaded image saved to {upload_path}")

        # Run YOLO predictions
        results = model.predict(source=upload_path, conf=0.25, save=True)
        logging.info(f"YOLO results: {results}")  # Inspect results for debugging

        # Retrieve the save directory from YOLO results
        if not results or not hasattr(results[0], 'save_dir'):
            logging.error("YOLO did not return a valid save directory.")
            return jsonify({"error": "Prediction failed"}), 500

        prediction_dir = results[0].save_dir
        logging.info(f"YOLO predictions saved in: {prediction_dir}")

        # Find the predicted image
        predicted_img = None
        for file_name in os.listdir(prediction_dir):
            if file_name.endswith(('.jpg', '.png', '.jpeg')):
                predicted_img = os.path.join(prediction_dir, file_name)
                break

        if not predicted_img:
            logging.error("No predicted image found in YOLO output.")
            return jsonify({"error": "Prediction failed"}), 500

        # Move the predicted image to the PREDICTION_FOLDER
        final_prediction_path = os.path.join(PREDICTION_FOLDER, os.path.basename(predicted_img))
        os.rename(predicted_img, final_prediction_path)
        logging.info(f"Predicted image moved to: {final_prediction_path}")

        # Add geolocation metadata
        add_geolocation(final_prediction_path, latitude, longitude)
    
        return jsonify({
            "image_url": f"/static/predictions/{os.path.basename(final_prediction_path)}?t={uuid.uuid4().hex}",
            "latitude": latitude,
            "longitude": longitude
        })


    except Exception as e:
        logging.exception("An error occurred during prediction.")
        return jsonify({"error": str(e)}), 500



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        image_path = filename

        if model is None:
            return jsonify({'error': 'Object detection model is unavailable.'}), 503
        try:
            # Perform object detection
            results = model(image_path)
            if not results or len(results) == 0:
                return jsonify({"error": "No detection results"}), 400

            result = results[0]
            labels = result.names

            # Count occurrences of each object
            object_counts = {}
            for box in result.boxes:
                label = labels[int(box.cls)]
                object_counts[label] = object_counts.get(label, 0) + 1

            # Save annotated image
            annotated_image = result.plot()
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"annotated_{file.filename}")
            Image.fromarray(annotated_image).save(output_path)

            return render_template('report.html', object_counts=object_counts)

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 400

# Serve Uploaded and Predicted Images
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/static/predictions/<filename>')
def prediction_file(filename):
    return send_from_directory(PREDICTION_FOLDER, filename)

@app.route('/map')
def map_page():
    # Get the latest predicted image
    predicted_files = sorted(
        os.listdir(PREDICTION_FOLDER),
        key=lambda x: os.path.getmtime(os.path.join(PREDICTION_FOLDER, x)),
        reverse=True
    )
    latest_image = predicted_files[0] if predicted_files else None

    
    

    return render_template('map.html', image=latest_image)



@app.errorhandler(405)
def method_not_allowed(e):
    user_name = session.get('name', 'User')
    return render_template('405.html', user_name=user_name), 405

@app.route('/uploaded_images')
def uploaded_images():
    # List all files in the UPLOAD_FOLDER directory
    image_files = os.listdir(UPLOAD_FOLDER)
    image_files = [file for file in image_files if file.endswith(('jpg', 'jpeg', 'png'))]  # Filter for image files
    return render_template('uploaded_images.html', images=image_files)

@app.route('/predicted_images')
def predicted_images():
    # List all files in the PREDICTION_FOLDER directory
    predicted_files = os.listdir(PREDICTION_FOLDER)
    predicted_files = [file for file in predicted_files if file.endswith(('jpg', 'jpeg', 'png'))]  # Filter for image files
    return render_template('predicted_images.html', images=predicted_files)



@app.route('/chat')
def chat_report():
    return render_template('chat.html')


# @app.route('/report_nlp', methods=['POST'])
# def generate_report_nlp():
#     data = request.get_json()  # Get JSON data from the request
#     if not data or 'object_counts' not in data:
#         return jsonify({'error': 'Invalid data'}), 400

#     object_counts = data['object_counts']
#     font_path = os.path.join(app.root_path, 'static', 'fonts', 'DejaVuSans.ttf')  # Correct font path
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.add_font('DejaVu', '', font_path, uni=True)
#     pdf.set_font("DejaVu", size=12)

#     # Title
#     pdf.cell(200, 10, txt="Image Analysis Report", ln=True, align='C')
#     pdf.ln(10)

#     # Detected Objects
#     pdf.cell(0, 10, txt="Objects Detected:", ln=True)
#     for label, count in object_counts.items():
#         pdf.cell(0, 10, txt=f"    • {label}: {count} detected", ln=True)
#     pdf.ln(10)

#     # Findings
#     pdf.cell(0, 10, txt="Findings:", ln=True)
#     findings_text = (
#         "Upon analyzing the provided image, the following objects were detected: " +
#         ", ".join(object_counts.keys()) + ". Each object was classified based on its "
#         "visual characteristics. This analysis provides insight into the distribution "
#         "of objects within the image."
#     )
#     pdf.multi_cell(0, 10, txt=findings_text)
#     pdf.ln(10)

#     # Conclusion
#     pdf.cell(0, 10, txt="Conclusion:", ln=True)
#     conclusion_text = (
#         "The report provides detailed insights into the objects detected in the image. "
#         "The analysis can be applied to areas such as surveillance, automated systems, and "
#         "environmental monitoring. Further investigation into the spatial distribution "
#         "and relationships among objects could enhance understanding and decision-making."
#     )
#     pdf.multi_cell(0, 10, txt=conclusion_text)

#     report_path = os.path.join(app.config['REPORT_FOLDER'], 'report.pdf')
#     pdf.output(report_path)

#     return send_file(report_path, as_attachment=True)

@app.route('/report_nlp', methods=['POST'])
def generate_report_nlp():
    # Extract and validate JSON data
    data = request.get_json()
    if not data or 'object_counts' not in data:
        return jsonify({'error': 'Invalid data'}), 400

    object_counts = data['object_counts']
    if not isinstance(object_counts, dict):
        return jsonify({'error': 'Invalid object_counts format'}), 400

    # Font setup: Use absolute path for the font
    font_path = r"C:/Users\DEEPANSHU/OneDrive\Desktop\sih\static/fonts\DejaVuSans.ttf"

    # Debugging: Check if the font path is accessible
    if not os.path.exists(font_path):
        return jsonify({'error': f"Font file not found at {font_path}"}), 500

    try:
        # PDF creation
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font('DejaVu', '', font_path, uni=True)
        pdf.set_font("DejaVu", size=12)

        # Title
        pdf.cell(200, 10, txt="Analysis Report", ln=True, align='C')
        pdf.ln(10)

        # Detected Objects
        pdf.cell(0, 10, txt="Objects Detected:", ln=True)
        for label, count in object_counts.items():
            pdf.cell(0, 10, txt=f"    • {label}: {count} detected", ln=True)
        pdf.ln(10)

        # Findings Section
        findings_text = (
            "Upon analyzing the provided image, the following objects were detected: " +
            ", ".join(object_counts.keys()) + ". Each object was classified based on its "
            "visual characteristics. This analysis provides insight into the distribution "
            "of objects within the image."
        )
        pdf.multi_cell(0, 10, txt="Findings:")
        pdf.multi_cell(0, 10, txt=findings_text)
        pdf.ln(10)

        # Conclusion Section
        conclusion_text = (
            "The report provides detailed insights into the objects detected in the image. "
            "The analysis can be applied to areas such as surveillance, automated systems, and "
            "environmental monitoring. Further investigation into the spatial distribution "
            "and relationships among objects could enhance understanding and decision-making."
        )
        pdf.multi_cell(0, 10, txt="Conclusion:")
        pdf.multi_cell(0, 10, txt=conclusion_text)

        # Save the PDF
        report_folder = app.config.get('REPORT_FOLDER', os.path.join(app.root_path, 'static', 'reports'))
        os.makedirs(report_folder, exist_ok=True)
        report_path = os.path.join(report_folder, 'report.pdf')
        pdf.output(report_path)
    except Exception as e:
        return jsonify({'error': f"Failed to generate report: {str(e)}"}), 500

    # Return the generated PDF
    return send_file(report_path, as_attachment=True)


@app.route('/download/previs')
def download_previs():
    return send_from_directory(
        directory='static/files', 
        path='FUSIONAI.pdf',
        as_attachment=True
    )

@app.route('/route')
def route():
    user_name = session.get('name', 'User')
    return render_template('route.html', user_name=user_name)

@app.route('/video')
def video_rep():
    return render_template('video.html')



@app.errorhandler(405)
def method_not_allowed(e):
    user_name = session.get('name', 'User')
    return render_template('405.html', user_name=user_name), 405


if __name__ == '__main__':
    app.run(debug=True)