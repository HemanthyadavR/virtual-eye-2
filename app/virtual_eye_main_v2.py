# app/virtual_eye_optimized.py
import cv2
import time
from collections import defaultdict
import threading
import queue
import os
import glob
import numpy as np
import asyncio
import io

# --- IMPORTS ---
import mediapipe as mp
import face_recognition
import google.generativeai as genai
import edge_tts
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play

# --- 1. PERFORMANCE OPTIMIZATIONS ---
# We will process a smaller image to speed everything up significantly.
PROCESSING_WIDTH = 480 
# Process every Nth frame. A higher number means less lag but slower reactions.
FRAME_PROCESSING_INTERVAL = 8 
# Use the faster 'hog' model for face detection on CPU instead of 'cnn'.
FACE_DETECTION_MODEL = "hog" 

# --- CONFIGURATION & TUNING ---
PRIORITY = {
    "person": 10, "car": 5, "bus": 5, "bicycle": 4, "motorbike": 4,
    "dog": 3, "cat": 3, "chair": 2, "table": 2, "phone": 1, "bottle": 1,
}
AI_NARRATION_INTERVAL = 10
MIN_NARRATION_GAP = 4
TOP_K = 5 

# --- AI & TTS SETUP ---
load_dotenv()
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=GOOGLE_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Gemini AI model configured successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not configure Gemini AI. Error: {e}")
    llm_model = None

VOICE = "en-US-JennyNeural"
speech_queue = queue.Queue()
playback_handle = None

# --- MEDIAPIPE POSE SETUP ---
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- UTILITY & SETUP FUNCTIONS ---
def load_known_faces(folder_path="known_faces"):
    known_face_encodings, known_face_names = [], []
    print(f"Loading known faces from '{folder_path}'...")
    if not os.path.exists(folder_path):
        print(f"Warning: '{folder_path}' directory not found.")
        return known_face_encodings, known_face_names
    for image_path in glob.glob(os.path.join(folder_path, "*.*")):
        try:
            name = os.path.splitext(os.path.basename(image_path))[0]
            face_image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(face_image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f" - Learned face: {name}")
        except Exception as e:
            print(f" - Error loading {os.path.basename(image_path)}: {e}")
    return known_face_encodings, known_face_names

def get_person_action(landmarks):
    """Analyzes pose landmarks to determine if a person is sitting or standing."""
    try:
        hip_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
        knee_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
        shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        threshold = (knee_y - shoulder_y) * 0.1 
        if hip_y > knee_y + threshold:
            return "is sitting"
        else:
            return "is standing"
    except:
        return None

def describe_scene_with_ai(scene_data):
    if not llm_model: return "AI model is not available."
    if not scene_data["objects"]: return "The scene appears to be clear."
    
    prompt = "You are an AI assistant for a visually impaired person. Describe the scene in a clear, concise, natural way. Here is the data from the camera:\n\n"
    object_descriptions = []
    for obj in scene_data["objects"]:
        desc = obj['label']
        if obj.get('action'):
            desc += f" {obj['action']}"
        desc += f" {obj['position']}"
        object_descriptions.append(desc)
            
    prompt += ", and ".join(object_descriptions) + "."
    prompt += "\n\nDescribe this scene in a single, fluid sentence."

    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini AI: {e}")
        return "There was an error describing the scene."

async def amain_tts_to_buffer(text_to_speak):
    buffer = io.BytesIO()
    communicate = edge_tts.Communicate(text_to_speak, VOICE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    buffer.seek(0)
    return buffer

def speech_worker():
    global playback_handle
    while True:
        text = speech_queue.get()
        if text is None: break
        print(f"AI Narrator: \"{text}\"")
        try:
            if playback_handle and playback_handle.is_playing():
                playback_handle.stop()
            audio_buffer = asyncio.run(amain_tts_to_buffer(text))
            audio_segment = AudioSegment.from_mp3(audio_buffer)
            playback_handle = play(audio_segment)
        except Exception as e:
            print(f"Error during TTS generation or playback: {e}")
        speech_queue.task_done()

def speak(text):
    while not speech_queue.empty():
        try: speech_queue.get_nowait()
        except queue.Empty: continue
    speech_queue.put(text)

def get_object_position(center_x, frame_width):
    zone_boundary_1 = frame_width / 3
    zone_boundary_2 = 2 * frame_width / 3
    if center_x < zone_boundary_1: return "on your left"
    elif center_x <= zone_boundary_2: return "in front of you"
    else: return "on your right"

# --- MAIN APPLICATION ---
def main():
    from ultralytics import YOLO 
    
    print("Starting Virtual Eye (Optimized)...")
    known_face_encodings, known_face_names = load_known_faces()
    
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()
    
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FATAL ERROR: Could not open webcam.")
        return

    last_confirmed_set = set()
    last_ai_narration_time = 0
    last_spoken_narration = ""
    frame_skip_counter = 0
    last_annotated_frame = None # <-- ADDED: To hold the last frame with drawings
    
    print("Application is running. Press ESC in the video window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        aspect_ratio = frame.shape[0] / frame.shape[1]
        processing_height = int(PROCESSING_WIDTH * aspect_ratio)
        resized_frame = cv2.resize(frame, (PROCESSING_WIDTH, processing_height))
        
        is_processing_frame = frame_skip_counter % FRAME_PROCESSING_INTERVAL == 0
        frame_skip_counter += 1

        if is_processing_frame:
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            yolo_results = model.predict(rgb_frame, conf=0.45, verbose=False)
            
            # --- FIX: Generate the annotated frame from YOLO results ---
            annotated_rgb = yolo_results[0].plot()
            last_annotated_frame = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            
            # --- Scene Analysis ---
            current_confirmed_objects = []
            persons_detected = []
            
            if hasattr(yolo_results[0].boxes, "cls"):
                all_boxes = yolo_results[0].boxes
                for i in range(len(all_boxes.cls)):
                    label = model.names[int(all_boxes.cls[i])]
                    box = all_boxes.xyxy[i]
                    x1, y1, x2, y2 = map(int, box)
                    
                    obj_data = {'label': label, 'box': (x1, y1, x2, y2)}
                    if label == 'person':
                        persons_detected.append(obj_data)
                    else:
                        current_confirmed_objects.append(obj_data)

            if persons_detected:
                persons_detected.sort(key=lambda p: (p['box'][2] - p['box'][0]) * (p['box'][3] - p['box'][1]), reverse=True)
                main_person = persons_detected[0]
                
                person_name = "a person"
                person_action = None

                face_locations = face_recognition.face_locations(rgb_frame, model=FACE_DETECTION_MODEL)
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        if True in matches:
                            person_name = known_face_names[matches.index(True)]
                            break
                
                pose_results = pose_estimator.process(rgb_frame)
                if pose_results.pose_landmarks:
                    person_action = get_person_action(pose_results.pose_landmarks)
                
                main_person['label'] = person_name
                main_person['action'] = person_action
                current_confirmed_objects.append(main_person)

            final_scene_objects = []
            for obj in current_confirmed_objects:
                x1, _, x2, _ = obj['box']
                center_x = (x1 + x2) / 2
                obj['position'] = get_object_position(center_x, resized_frame.shape[1])
                final_scene_objects.append(obj)

            current_confirmed_set = set(obj['label'] for obj in final_scene_objects)
            time_since_last_narration = time.time() - last_ai_narration_time
            
            should_call_ai = False
            if current_confirmed_set != last_confirmed_set and time_since_last_narration > MIN_NARRATION_GAP:
                should_call_ai = True
            elif time_since_last_narration > AI_NARRATION_INTERVAL and current_confirmed_set:
                should_call_ai = True

            if should_call_ai:
                final_scene_objects.sort(key=lambda x: PRIORITY.get(x['label'], 0), reverse=True)
                scene_data = {"objects": final_scene_objects[:TOP_K]}
                narration = describe_scene_with_ai(scene_data)
                
                if narration and narration != last_spoken_narration and "error" not in narration.lower():
                    speak(narration)
                    last_spoken_narration = narration

                last_confirmed_set = current_confirmed_set
                last_ai_narration_time = time.time()
        
        # --- FIX: Display the last annotated frame ---
        # This makes the visuals consistent with the processed information.
        display_frame = frame # Default to raw camera feed
        if last_annotated_frame is not None:
            # Resize the smaller annotated frame back to the original size for display
            display_frame = cv2.resize(last_annotated_frame, (frame.shape[1], frame.shape[0]))
        
        cv2.imshow("Virtual Eye - Press ESC to quit", display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("Shutting down...")
    pose_estimator.close()
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)
    speech_thread.join()

if __name__ == "__main__":
    main()

