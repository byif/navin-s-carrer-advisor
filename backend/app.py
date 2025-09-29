from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleWARE
from fastapi.templating import Jinja2Templates
import pdfplumber
import traceback
import base64
import numpy as np
import cv2
from deepface import DeepFace
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
import dlib
from math import degrees, atan2

# === Database Configuration (Stays the same) ===
DATABASE_URL = "sqlite:///resumes.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ResumeRecord(Base):
    __tablename__ = "resume_records"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    extracted_text = Column(String)
    predicted_career = Column(String)
    score = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)

# === FastAPI App and Middleware ===
app = FastAPI()
app.add_middleware(
    CORSMiddleWARE,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# === Facial Landmark and Head Pose Setup ===
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print(f"Error loading dlib shape predictor: {e}")
    predictor = None

# Global variables for instantaneous tracking (NOTE: Vercel will reset these frequently)
suspicious_activity_count = 0 
suspicious_threshold = 1
suspicious_frame_count = 0

# === Head Pose Estimation Function (Stays the same) ===
def get_head_pose_status(frame):
    global suspicious_frame_count, suspicious_activity_count
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    
    focus_status = "Looking at camera"

    if len(faces) > 0 and predictor:
        face = faces[0]
        landmarks = predictor(gray, face)
        
        nose_tip = landmarks.part(30)
        chin = landmarks.part(8)
        left_eye_left = landmarks.part(36)
        right_eye_right = landmarks.part(45)
        
        yaw = degrees(atan2(nose_tip.x - ((left_eye_left.x + right_eye_right.x) / 2), nose_tip.y))
        pitch = degrees(atan2(nose_tip.y - chin.y, nose_tip.x - chin.x))
        
        yaw_threshold = 3          
        pitch_lower_bound = -120   
        pitch_upper_bound = -80    

        is_suspicious = False
        
        if abs(yaw) > yaw_threshold:
            is_suspicious = True
            if yaw < 0:
                focus_status = "Looking left"
            else:
                focus_status = "Looking right"
        
        if pitch < pitch_lower_bound or pitch > pitch_upper_bound:
            is_suspicious = True
            if pitch < pitch_lower_bound:
                focus_status = "Looking up"
            else:
                focus_status = "Looking down"
        
        if is_suspicious:
            suspicious_frame_count += 1
        else:
            suspicious_frame_count = 0
            
        if suspicious_frame_count >= suspicious_threshold:
            suspicious_activity_count += 1
            suspicious_frame_count = 0
            
    else:
        focus_status = "No face detected"
        suspicious_frame_count = 0

    return focus_status, suspicious_activity_count

# === NEW: HTTP POST Endpoint for Serverless Analysis ===
# Vercel will map your frontend request to this function.
@app.post("/api/analyze-frame")
async def analyze_frame_http(request: Request):
    """
    Handles a single, high-frequency HTTP request for video frame analysis.
    """
    try:
        data = await request.json()
        encoded_data = data.get('frame_data', '').split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")

        # 1. Emotion Analysis
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='ssd')
            emotion = results[0]['dominant_emotion'] if results else 'No face detected'
        except Exception:
            emotion = 'Processing error'

        # 2. Focus Status (Uses global, which is fine for instantaneous reading)
        focus_status, count = get_head_pose_status(frame)

        return {
            'emotion': emotion,
            'focus_status': focus_status,
            'suspicious_count': count
        }

    except HTTPException as e:
        # Pass through specific HTTP exceptions
        raise e
    except Exception as e:
        # Catch all other exceptions
        return JSONResponse(status_code=500, content={'error': f'Serverless processing failed: {e}'})

# === Resume-related routes (Stays the same) ===
def is_valid_resume(text):
    resume_keywords = [
        'education', 'experience', 'skills', 'projects',
        'certifications', 'internship', 'objective',
        'b.tech', 'developer', 'python', 'machine learning',
        'msc', 'bsc', 'career', 'summary'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in resume_keywords)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
async def upload_resume(resume: UploadFile = File(...)):
    db = SessionLocal()
    try:
        if not resume.filename.lower().endswith('.pdf'):
            return JSONResponse(status_code=400, content={'error': 'Invalid file type. Please upload a PDF.'})

        contents = await resume.read()
        
        temp_file_path = f"temp_{resume.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(contents)
            
        with pdfplumber.open(temp_file_path) as pdf:
            text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())

        os.remove(temp_file_path)

        if not text.strip():
            return JSONResponse(status_code=400, content={'error': 'No readable text found in the PDF.'})
        
        if not is_valid_resume(text):
            return JSONResponse(status_code=400, content={'error': 'This file does not appear to be a valid resume.'})

        from model.predictor import predict_careers_from_text
        predictions = predict_careers_from_text(text)
        predicted_career = predictions[0] if predictions else 'Unknown'
        score = 100 if predicted_career != 'Unknown' else 0

        record = ResumeRecord(
            filename=resume.filename,
            extracted_text=text,
            predicted_career=predicted_career,
            score=score
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return JSONResponse(status_code=200, content={
            'success': True,
            'predicted_career': predicted_career,
            'all_predictions': predictions,
            'score': score
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': f'Internal server error: {e}'})
    finally:
        db.close()

@app.get("/records")
async def get_all_records():
    db = SessionLocal()
    try:
        records = db.query(ResumeRecord).order_by(ResumeRecord.uploaded_at.desc()).all()
        return [
            {
                'id': r.id,
                'filename': r.filename,
                'predicted_career': r.predicted_career,
                'score': r.score,
                'uploaded_at': r.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')
            } for r in records
        ]
    finally:
        db.close()

# === The old @app.websocket("/ws") function has been removed. ===
