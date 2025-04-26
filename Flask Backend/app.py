import os
import zipfile
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from easyocr import Reader
import cv2
import pandas as pd
from fuzzywuzzy import fuzz
import re
from datetime import datetime
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from bson import ObjectId

# Flask application setup
app = Flask(__name__)

app.config['MONGO_URI'] = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/aadhaar_verification')
mongo_client = MongoClient(app.config['MONGO_URI'])
db = mongo_client.get_database() # This will now use 'aadhaar_verification' database

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
            # Debug: Print Excel file path
            print(f"\n[DEBUG] Reading Excel file from: {excel_path}")
            
            # Check if file exists
            if not os.path.exists(excel_path):
                return [{
                    "status": "Error",
                    "reason": "Excel file not found at specified path",
                    "file_path": excel_path
                }]

            # Read Excel file
            try:
                excel_data = pd.read_excel(excel_path)
                print("[DEBUG] Excel file read successfully")
                print(f"[DEBUG] Columns found: {excel_data.columns.tolist()}")
            except Exception as e:
                return [{
                    "status": "Error",
                    "reason": f"Failed to read Excel file: {str(e)}",
                    "file_path": excel_path
                }]

            # Check required columns
            required_columns = ['UID', 'Name']
            missing_columns = [col for col in required_columns if col not in excel_data.columns]
            if missing_columns:
                return [{
                    "status": "Error",
                    "reason": f"Missing required columns in Excel: {', '.join(missing_columns)}",
                    "available_columns": excel_data.columns.tolist()
                }]

            uid = fields.get("uid")
            extracted_name = fields.get("name", "N/A")
            extracted_address = fields.get("address")

            if not uid:
                return [{"status": "Rejected", "reason": "UID not found in image."}]

            uid_cleaned = self.clean_uid(uid)
            best_match = None
            highest_score = 0

            # Debug: Print first few UIDs from Excel
            print(f"[DEBUG] First 5 UIDs from Excel: {excel_data['UID'].head().tolist()}")
            print(f"[DEBUG] Extracted UID: {uid_cleaned}")

            for _, row in excel_data.iterrows():
                # Skip rows with empty UID
                if pd.isna(row.get("UID")):
                    continue
                    
                excel_uid_cleaned = self.clean_uid(str(row.get("UID", "")))

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
                
                # Change this in the compare_with_excel method
                return [{
                    "status": "Rejected",
                    "reason": "No matching UID found in database",
                    "Extracted Name": extracted_name,
                    "Extracted UID": uid_cleaned,
                    "UID Match Score": 0,  # Changed from None to 0
                    "Name Match Score": 0,
                    "Address Match Score": 0,
                    "Overall Match Score": 0
                }]
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
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
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

# Initialize Aadhaar verification system
verifier = AadhaarVerificationSystem(
   upload_folder = os.environ.get('UPLOAD_FOLDER', 'uploads'),
   classifier_path = os.environ.get('CLASSIFIER_PATH', 'models/classifier.pt'),
   detector_path = os.environ.get('DETECTOR_PATH', 'models/detector.pt')
)

@app.route('/analytics')
def analytics_dashboard():
    try:
        # 1. Overall Verification Statistics
        total_files = db.file_details.count_documents({})
        processed_files = db.file_details.count_documents({"status": "Processed"})
        
        # 2. Verification Status Distribution
        verification_stats = list(db.verification.aggregate([
            {"$group": {
                "_id": "$status",
                "count": {"$sum": 1},
                "percentage": {
                    "$avg": {
                        "$cond": [
                            {"$eq": [1, 1]},  # Always true to calculate percentage
                            {"$multiply": [{"$divide": [100, total_files]}, 1]},
                            0
                        ]
                    }
                }
            }},
            {"$project": {
                "status": "$_id",
                "count": 1,
                "percentage": {"$round": ["$percentage", 2]}
            }}
        ]))
        
        # 3. Match Score Analysis
        match_score_analysis = db.extracted_details.aggregate([
            {"$group": {
                "_id": None,
                "min_score": {"$min": "$overall_match_score"},
                "max_score": {"$max": "$overall_match_score"},
                "avg_score": {"$avg": "$overall_match_score"},
                "score_deviation": {"$stdDevPop": "$overall_match_score"}
            }}
        ]).next()
        
        # 4. Score Range Distribution
        score_ranges = list(db.extracted_details.aggregate([
            {"$bucket": {
                "groupBy": "$overall_match_score",
                "boundaries": [0, 50, 70, 85, 100],
                "default": "Unknown",
                "output": {
                    "count": {"$sum": 1},
                    "percentage": {
                        "$avg": {
                            "$multiply": [{"$divide": [100, total_files]}, 1]
                        }
                    }
                }
            }},
            {"$project": {
                "score_range": {
                    "$switch": {
                        "branches": [
                            {"case": {"$lt": ["$_id", 50]}, "then": "Low Match (0-50)"},
                            {"case": {"$and": [
                                {"$gte": ["$_id", 50]},
                                {"$lt": ["$_id", 70]}
                            ]}, "then": "Moderate Match (50-70)"},
                            {"case": {"$and": [
                                {"$gte": ["$_id", 70]},
                                {"$lt": ["$_id", 85]}
                            ]}, "then": "Good Match (70-85)"},
                            {"case": {"$gte": ["$_id", 85]}, "then": "Excellent Match (85-100)"}
                        ],
                        "default": "Unknown"
                    }
                },
                "count": 1,
                "percentage": {"$round": ["$percentage", 2]}
            }}
        ]))
        
        # 5. Monthly Processing Trend
        monthly_trend = list(db.verification.aggregate([
            {"$project": {
                "month": {"$dateToString": {"format": "%Y-%m", "date": "$processed_at"}},
                "status": 1
            }},
            {"$group": {
                "_id": {"month": "$month", "status": "$status"},
                "count": {"$sum": 1},
                "percentage": {
                    "$avg": {
                        "$multiply": [{"$divide": [100, total_files]}, 1]
                    }
                }
            }},
            {"$project": {
                "month": "$_id.month",
                "status": "$_id.status",
                "count": 1,
                "percentage": {"$round": ["$percentage", 2]}
            }},
            {"$sort": {"month": 1}}
        ]))
        
        # 6. Field-level Matching Analysis
        field_matching = db.extracted_details.aggregate([
            {"$group": {
                "_id": None,
                "avg_name_match": {"$avg": "$name_match_score"},
                "avg_address_match": {"$avg": "$address_match_score"},
                "avg_uid_match": {"$avg": "$uid_match_score"}
            }}
        ]).next()
        
        # Prepare data for template
        analytics_data = {
            'total_files': total_files,
            'processed_files': processed_files,
            'verification_stats': {stat['status']: stat for stat in verification_stats},
            # Before passing data to template, ensure no null values
            'match_score_analysis':{
                'min_score': match_score_analysis.get('min_score') or 0,
                'max_score': match_score_analysis.get('max_score') or 0,
                'avg_score': round(match_score_analysis.get('avg_score') or 0, 2),
                'score_deviation': round(match_score_analysis.get('score_deviation') or 0, 2)
            },
            'score_ranges': {range['score_range']: range for range in score_ranges},
            'monthly_trend': monthly_trend,
            'field_matching': {
                'avg_name_match': round(field_matching['avg_name_match'], 2),
                'avg_address_match': round(field_matching['avg_address_match'], 2),
                'avg_uid_match': round(field_matching['avg_uid_match'], 2)
            }
        }
        
        return render_template('analytics.html', analytics_data=analytics_data)
    
    except Exception as e:
        return jsonify({"error": f"Analytics generation error: {str(e)}"}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        if 'zipfile' not in request.files or 'excelfile' not in request.files:
            return jsonify({"error": "Both files are required."}), 400

        zip_file = request.files['zipfile']
        excel_file = request.files['excelfile']

        zip_filename = secure_filename(zip_file.filename)
        excel_filename = secure_filename(excel_file.filename)
        
        zip_path = os.path.join(verifier.upload_folder, zip_filename)
        excel_path = os.path.join(verifier.upload_folder, excel_filename)

        os.makedirs(verifier.upload_folder, exist_ok=True)
        zip_file.save(zip_path)
        excel_file.save(excel_path)

        try:
            results = verifier.process_zip_file(zip_path, excel_path)

            # Store file details in MongoDB
            db.file_details.insert_one({
                "filename": zip_filename,
                "uploaded_at": datetime.utcnow(),
                "processed_at": datetime.utcnow(),
                "status": "Processed"
            })

            # Store extracted details and verification results
            for result in results:
                if 'match_results' in result:
                    for match in result['match_results']:
                        overall_score = match.get('Overall Match Score')
                        
                        if overall_score is not None and overall_score >= 70:
                            # Store extracted details
                            db.extracted_details.insert_one({
                                "filename": result['filename'],
                                "name": match.get('Extracted Name'),
                                "uid": match.get('UID'),
                                "address": match.get('Address Reference'),
                                "name_match_score": match.get('Name Match Score'),
                                "address_match_score": match.get('Address Match Score'),
                                "uid_match_score": match.get('UID Match Score'),
                                "overall_match_score": match.get('Overall Match Score'),
                                "status": match.get('status'),
                                "reason": match.get('reason'),
                                "processed_at": datetime.utcnow()
                            })

                            # Store verification status
                            db.verification.insert_one({
                                "filename": result['filename'],
                                "status": "Accepted",
                                "processed_at": datetime.utcnow()
                            })
                        elif overall_score is not None and overall_score < 70:
                            db.verification.insert_one({
                                "filename": result['filename'],
                                "status": "Rejected",
                                "reason": match.get('reason', "Low match score"),
                                "processed_at": datetime.utcnow()
                            })

            return jsonify({"results": results})
        except Exception as e:
            return jsonify({"error": f"Error during processing: {str(e)}"}), 500
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(excel_path):
                os.remove(excel_path)
    else:
        return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
