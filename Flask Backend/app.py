import os
import zipfile
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from easyocr import Reader
import cv2
import pandas as pd
from fuzzywuzzy import fuzz
import re
from datetime import datetime

# Flask application setup
app = Flask(__name__)

# Set up PostgreSQL URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/fraud_detection_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class AadhaarVerificationSystem:
    def __init__(self, upload_folder, classifier_path, detector_path):
        self.upload_folder = upload_folder
        self.extract_folder = os.path.join(upload_folder, 'extracted_files')
        os.makedirs(self.extract_folder, exist_ok=True)

        # Initialize models
        self.classifier = YOLO(classifier_path)
        self.detector = YOLO(detector_path)
        self.ocr_reader = Reader(['en'])

    def clean_text(self, text):
        return ''.join(e for e in str(text) if e.isalnum()).lower()

    def clean_uid(self, uid):
        return ''.join(filter(str.isdigit, str(uid)))

    def clean_address(self, address):
        address = address.lower()
        address = re.sub(r'\s+', ' ', address)  # Remove extra spaces
        address = re.sub(r'(marg|lane|township|block|street)', '', address)  # Remove common terms
        return address

    def is_aadhaar_card(self, image_path):
        try:
            prediction = self.classifier.predict(source=image_path)
            class_index = prediction[0].probs.top1
            class_name = prediction[0].names[class_index]
            confidence = prediction[0].probs.top1conf.item()

            # Debug print
            print(f"Classification: {class_name}, Confidence: {confidence}")

            return class_name.lower() == "aadhar" or class_name.lower() == "aadhaar", confidence
        except Exception as e:
            print(f"Classification error: {str(e)}")
            return False, 0

    def detect_fields(self, image_path):
        try:
            results = self.detector(image_path)[0]
            fields = {}

            image = cv2.imread(image_path)
            for box in results.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.detector.names[class_id]
                coords = box.xyxy[0].cpu().numpy().astype(int)

                if conf > 0.5:
                    x1, y1, x2, y2 = coords
                    cropped_roi = image[y1:y2, x1:x2]

                    # Preprocess for OCR
                    gray = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    text = self.ocr_reader.readtext(thresh, detail=0)
                    fields[label] = ' '.join(text) if text else None

            return fields
        except Exception as e:
            print(f"Field detection error: {str(e)}")
            return {}

    def match_names(self, extracted_name, excel_name):
        name_score = fuzz.ratio(self.clean_text(extracted_name), self.clean_text(excel_name))

        if name_score < 100:
            extracted_parts = extracted_name.split()
            excel_parts = excel_name.split()
            if len(extracted_parts) > 1 and len(excel_parts) > 1:
                # Allow abbreviation of first name
                if fuzz.ratio(extracted_parts[0][0], excel_parts[0][0]) > 90:
                    name_score = 90

        if name_score < 100:
            extracted_parts = extracted_name.split()
            excel_parts = excel_name.split()
            if len(extracted_parts) == 2 and len(excel_parts) > 2:
                if extracted_parts[0] == excel_parts[0] and extracted_parts[1] == excel_parts[2]:
                    name_score = 90

        if name_score < 100:
            if any(part in extracted_name for part in excel_name.split()):
                name_score = 90

        if name_score < 100:
            if sorted(extracted_name.split()) == sorted(excel_name.split()):
                name_score = 90

        if name_score < 100:
            for part in excel_name.split():
                if len(part) == 1 and part.lower() == extracted_name[0].lower():
                    name_score = 90
                    break

        return name_score

    def match_addresses(self, extracted_address, row):
        address_score = 0
        address_components = ['Street Road Name', 'City', 'State', 'PINCODE']
        full_address = ' '.join([str(row[comp]) for comp in address_components if row[comp]])

        cleaned_extracted_address = self.clean_address(extracted_address)
        cleaned_full_address = self.clean_address(full_address)

        address_score = fuzz.partial_ratio(cleaned_extracted_address, cleaned_full_address)

        extracted_pincode = re.sub(r'\D', '', extracted_address)
        if extracted_pincode == row['PINCODE']:
            address_score = 100

        return address_score, full_address

    def compare_with_excel(self, fields, excel_path):
        try:
            excel_data = pd.read_excel(excel_path)
            uid = fields.get("uid")
            extracted_name = fields.get("name", "N/A")
            extracted_address = fields.get("address")

            if not uid:
                return [{"status": "Rejected", "reason": "UID not found in image."}]  # For rejected files

            uid_cleaned = self.clean_uid(uid)
            best_match = None
            highest_score = 0

            for _, row in excel_data.iterrows():
                excel_uid_cleaned = self.clean_uid(row.get("UID", ""))

                name_score = 0
                if extracted_name != "N/A":
                    name_score = self.match_names(extracted_name, row.get("Name", ""))

                address_score = 0
                full_address = None
                if extracted_address:
                    address_score, full_address = self.match_addresses(extracted_address, row)

                uid_score = fuzz.ratio(uid_cleaned, excel_uid_cleaned)

                if extracted_name != "N/A" and extracted_address:
                    overall_score = (name_score + address_score + uid_score) / 3
                elif extracted_name != "N/A":
                    overall_score = (name_score + uid_score) / 2
                elif extracted_address:
                    overall_score = (address_score + uid_score) / 2
                else:
                    overall_score = uid_score

                status = "Accepted" if overall_score >= 70 else "Rejected"

                if overall_score > highest_score:
                    highest_score = overall_score
                    best_match = {
                        "SrNo": row.get("SrNo"),
                        "Name": row.get("Name"),
                        "Extracted Name": extracted_name,
                        "UID": row.get("UID"),
                        "Address Match Score": address_score if extracted_address else None,
                        "Address Reference": full_address,
                        "Name Match Score": name_score if extracted_name != "N/A" else None,
                        "UID Match Score": uid_score,
                        "Overall Match Score": overall_score,
                        "status": status,
                        "reason": "Matching scores calculated."
                    }

            if best_match is None:
                return [{"status": "Rejected", "reason": "No matching record found."}]  # For rejected files

            return [best_match]

        except Exception as e:
            print(f"Excel comparison error: {str(e)}")
            return [{"status": "Error", "reason": str(e)}]

    def process_zip_file(self, zip_path, excel_path):
        results = []
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_folder)

            for root, _, files in os.walk(self.extract_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process image files only
                        image_path = os.path.join(root, file)
                        is_aadhar, confidence = self.is_aadhaar_card(image_path)

                        if is_aadhar:
                            fields = self.detect_fields(image_path)
                            fields["filename"] = file
                            match_results = self.compare_with_excel(fields, excel_path)
                            results.append({
                                'filename': file,
                                'is_aadhar': is_aadhar,
                                'confidence': confidence,
                                'fields': fields,
                                'match_results': match_results
                            })
                        else:
                            results.append({
                                'filename': file,
                                'is_aadhar': is_aadhar,
                                'confidence': confidence,
                                'reason': "Not an Aadhaar card."
                            })
            return results

        except Exception as e:
            print(f"Zip processing error: {str(e)}")
            raise


# Database model for storing file details
class FileDetails(db.Model):
    __tablename__ = 'file_details'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(50), nullable=False)


# Database model for storing extracted details and matching scores
class ExtractedDetails(db.Model):
    __tablename__ = 'extracted_details'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255))
    uid = db.Column(db.String(255))
    address = db.Column(db.String(255))
    name_match_score = db.Column(db.Integer)
    address_match_score = db.Column(db.Integer)
    uid_match_score = db.Column(db.Integer)
    overall_match_score = db.Column(db.Integer)
    status = db.Column(db.String(50))
    reason = db.Column(db.String(255))
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)


# Database model for storing verification results (Accepted/Rejected status)
class Verification(db.Model):
    __tablename__ = 'verification'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)


# Create tables automatically if they don't exist
with app.app_context():
    db.create_all()

# Initialize Aadhaar verification system
verifier = AadhaarVerificationSystem(
    upload_folder="C:/Users/DEIVA RAJA/Documents/Infosys_Fraud_Detection/uploads",
    classifier_path="C:/Users/DEIVA RAJA/runs/classify/train6/weights/best.pt",
    detector_path="C:/Users/DEIVA RAJA/runs/detect/train6/weights/best.pt"
)

@app.route('/analytics')
def analytics_dashboard():
    try:
        # 1. Overall Verification Statistics
        total_files = FileDetails.query.count()
        processed_files = FileDetails.query.filter_by(status="Processed").count()
        
        # 2. Verification Status Distribution with more detailed breakdown
        verification_stats = db.session.query(
            Verification.status, 
            db.func.count(Verification.id).label('count'),
            db.func.round(100.0 * db.func.count(Verification.id) / total_files, 2).label('percentage')
        ).group_by(Verification.status).all()
        
        # 3. Enhanced Match Score Analysis
        match_score_analysis = db.session.query(
            db.func.min(ExtractedDetails.overall_match_score).label('min_score'),
            db.func.max(ExtractedDetails.overall_match_score).label('max_score'),
            db.func.avg(ExtractedDetails.overall_match_score).label('avg_score'),
            db.func.stddev(ExtractedDetails.overall_match_score).label('score_deviation')
        ).first()
        
        # 4. Detailed Score Range Distribution
        score_ranges = db.session.query(
            db.case(
                (ExtractedDetails.overall_match_score < 50, 'Low Match (0-50)'),
                (ExtractedDetails.overall_match_score.between(50, 70), 'Moderate Match (50-70)'),
                (ExtractedDetails.overall_match_score.between(70, 85), 'Good Match (70-85)'),
                (ExtractedDetails.overall_match_score >= 85, 'Excellent Match (85-100)')
            , else_='Unknown').label('score_range'),
            db.func.count(ExtractedDetails.id).label('count'),
            db.func.round(100.0 * db.func.count(ExtractedDetails.id) / total_files, 2).label('percentage')
        ).group_by('score_range').all()
        
        # 5. Enhanced Monthly Processing Trend
        monthly_trend = db.session.query(
            db.func.date_trunc('month', Verification.processed_at).label('month'),
            Verification.status,
            db.func.count(Verification.id).label('count'),
            db.func.round(100.0 * db.func.count(Verification.id) / total_files, 2).label('percentage')
        ).group_by('month', Verification.status).order_by('month').all()
        
        # 6. Field-level Matching Analysis
        field_matching = db.session.query(
            db.func.avg(ExtractedDetails.name_match_score).label('avg_name_match'),
            db.func.avg(ExtractedDetails.address_match_score).label('avg_address_match'),
            db.func.avg(ExtractedDetails.uid_match_score).label('avg_uid_match')
        ).first()
        
        # Prepare comprehensive data for template rendering
        analytics_data = {
            'total_files': total_files,
            'processed_files': processed_files,
            'verification_stats': {
                row.status: {
                    'count': row.count, 
                    'percentage': row.percentage
                } for row in verification_stats
            },
            'match_score_analysis': {
                'min_score': match_score_analysis.min_score,
                'max_score': match_score_analysis.max_score,
                'avg_score': round(match_score_analysis.avg_score, 2),
                'score_deviation': round(match_score_analysis.score_deviation, 2)
            },
            'score_ranges': {
                row.score_range: {
                    'count': row.count, 
                    'percentage': row.percentage
                } for row in score_ranges
            },
            'monthly_trend': [
                {
                    'month': row.month.strftime('%Y-%m'),
                    'status': row.status,
                    'count': row.count,
                    'percentage': row.percentage
                } for row in monthly_trend
            ],
            'field_matching': {
                'avg_name_match': round(field_matching.avg_name_match, 2),
                'avg_address_match': round(field_matching.avg_address_match, 2),
                'avg_uid_match': round(field_matching.avg_uid_match, 2)
            }
        }
        
        return render_template('analytics.html', analytics_data=analytics_data)
    
    except Exception as e:
        return jsonify({"error": f"Analytics generation error: {str(e)}"}), 500
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'zipfile' not in request.files or 'excelfile' not in request.files:
        return jsonify({"error": "Both files are required."}), 400

    zip_file = request.files['zipfile']
    excel_file = request.files['excelfile']

    zip_path = os.path.join(verifier.upload_folder, zip_file.filename)
    excel_path = os.path.join(verifier.upload_folder, excel_file.filename)

    os.makedirs(verifier.upload_folder, exist_ok=True)
    zip_file.save(zip_path)
    excel_file.save(excel_path)

    try:
        results = verifier.process_zip_file(zip_path, excel_path)

        # Store file details in the database
        new_file = FileDetails(filename=zip_file.filename, status="Processed")
        db.session.add(new_file)
        db.session.commit()

        # Store extracted details in the new table (only accepted files)
        for result in results:
            if 'match_results' in result:
                for match in result['match_results']:
                    overall_score = match.get('Overall Match Score')
                    
                    # Only store details if the overall match score is >= 70 (Accepted files)
                    if overall_score >= 70:
                        extracted_details = ExtractedDetails(
                            filename=result['filename'],
                            name=match.get('Extracted Name'),
                            uid=match.get('UID'),
                            address=match.get('Address Reference'),
                            name_match_score=match.get('Name Match Score'),
                            address_match_score=match.get('Address Match Score'),
                            uid_match_score=match.get('UID Match Score'),
                            overall_match_score=match.get('Overall Match Score'),
                            status=match.get('status'),
                            reason=match.get('reason')
                        )
                        db.session.add(extracted_details)

                        # Add verification status (only Accepted)
                        verification_entry = Verification(
                            filename=result['filename'],
                            status="Accepted"
                        )
                        db.session.add(verification_entry)

        db.session.commit()

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": f"Error during processing: {str(e)}"}), 500
    finally:
        # Cleanup files
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(excel_path):
            os.remove(excel_path)


if __name__ == '__main__':
    app.run(debug=True)
