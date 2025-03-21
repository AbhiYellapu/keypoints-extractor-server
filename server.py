import os
from flask import Flask, request, jsonify
from moviepy import VideoFileClip
import speech_recognition as sr
from transformers import pipeline
import torch
from flask_cors import CORS
from utils import split_audio, convert_audio_to_text, extract_audio

def extract_keypoints(video_file_path, max_length=None, min_length=50):
    try:
        # Load the video and extract audio
        # video = VideoFileClip(video_file_path)
        # audio_path = "./uploads/extracted_audio.wav"
        # video.audio.write_audiofile(audio_path, codec="pcm_s16le")  # Ensure it's in WAV format

        # Initialize recognizer
        # recognizer = sr.Recognizer()

        # # Process the extracted audio file
        # with sr.AudioFile(audio_path) as source:
        #     recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        #     audio = recognizer.record(source)  # Record the entire audio

        # # Recognize speech using Google's speech recognition API
        # text = recognizer.recognize_google(audio)
        
        audio_path, _ = extract_audio(video_file_path)
        
        chunks = split_audio(audio_path, chunk_length=60)  # Split into 60-sec chunks
        text = convert_audio_to_text(chunks)

        # Initialize the summarization pipeline
        device = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
        
        max_length = max_length if max_length else len(text) // 2  # Default max_length = len(text)/
        print(max_length)
        min_length = min_length if min_length else 50  # Default min_length = 50

        # Handle long text by chunking
        chunk_size = 1000
        if len(text) > chunk_size:
            text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            summary = " ".join(
                summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
                for chunk in text_chunks
            )
        else:
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

        # Cleanup extracted audio
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return {"status": "success", "text": text, "keypoints": summary}

    except sr.UnknownValueError:
        return {"status": "error", "message": "Google Speech Recognition could not understand the audio"}
    except sr.RequestError as e:
        return {"status": "error", "message": f"Google Speech Recognition request failed: {e}"}
    except FileNotFoundError:
        return {"status": "error", "message": "The specified video file was not found"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

app = Flask(__name__)
CORS(app)

# Path to save uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route for the home page
@app.route("/healthcheck")
def home():
    return jsonify({
        "message": "Welcome to the AI Model API!",
        "status": "healthy",
        "try": "Post a video file to /api/keypoints-extractor"
    })

# Route for uploading and processing video files
@app.route('/api/keypoints-extractor', methods=['POST'])
def keypoints_extractor():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Extract keypoints from video file
    response = extract_keypoints(file_path, 100, 50)
    
    if response["status"] == "success":
        keypoints = response["keypoints"]
        text = response["text"]
        return jsonify({"keypoints": keypoints, "text": text}), 200
    elif response["status"] == "error":
        return jsonify({"error": response["message"]}), 500
    else:
        return jsonify({"error": "Unknown error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
