from pydub import AudioSegment
import speech_recognition as sr
import os

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
def split_audio(audio_path, chunk_length=60):
    audio = AudioSegment.from_wav(audio_path)
    audio_length_ms = len(audio)  # Get duration in milliseconds
    chunks = []

    for start_ms in range(0, audio_length_ms, chunk_length * 1000):
        end_ms = min(start_ms + (chunk_length * 1000), audio_length_ms)
        chunk = audio[start_ms:end_ms]
        chunk_filename = f"chunk_{start_ms//1000}.wav"
        chunk.export(chunk_filename, format="wav")
        chunks.append(chunk_filename)

    return chunks

def convert_audio_to_text(audio_chunks):
    recognizer = sr.Recognizer()
    full_text = ""

    for chunk in audio_chunks:
        with sr.AudioFile(chunk) as source:
            recognizer.adjust_for_ambient_noise(source)  # Adjust for noise
            audio = recognizer.record(source)  # Record chunk
            try:
                text = recognizer.recognize_google(audio)  # Convert to text
                full_text += text + " "  # Append text
            except sr.UnknownValueError:
                print(f"Google Speech Recognition could not understand {chunk}")
            except sr.RequestError as e:
                print(f"Google Speech Recognition request failed: {e}")

        os.remove(chunk)  # Delete processed chunk

    return full_text.strip()

def extract_audio(video_or_audio_path):
    """
    Extracts audio if the file is a video, otherwise converts MP3 to WAV.
    """
    file_ext = os.path.splitext(video_or_audio_path)[1].lower()
    
    if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:  # Video file
        from moviepy import VideoFileClip
        video = VideoFileClip(video_or_audio_path)
        audio_path = os.path.join(UPLOAD_FOLDER, "extracted_audio.wav")
        video.audio.write_audiofile(audio_path, codec="pcm_s16le")
    elif file_ext in ['.mp3', '.wav']:  # Audio file
        audio_path = os.path.join(UPLOAD_FOLDER, "extracted_audio.wav")
        audio = AudioSegment.from_file(video_or_audio_path)
        audio.export(audio_path, format="wav")
    else:
        return None, "Unsupported file format"
    
    return audio_path, None