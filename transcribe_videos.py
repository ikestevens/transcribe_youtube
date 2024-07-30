import csv
import os
import re
import yt_dlp
from pydub import AudioSegment
import whisper
from datetime import timedelta
import warnings
from tqdm import tqdm
import textwrap

# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def download_audio(url, filename):
    temp_filename = filename.replace('.mp3', '') + ".temp.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': temp_filename.replace('.mp3', ''),  # Use the specified temp filename without .mp3
        'ffmpeg_location': '/usr/local/bin',  # Specify the path to ffmpeg if needed
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Rename temp file to final filename
    if os.path.exists(temp_filename):
        os.rename(temp_filename, filename)

    # Get the duration of the audio file
    audio = AudioSegment.from_file(filename, format="mp3")
    duration_minutes = len(audio) / 60000  # Convert milliseconds to minutes
    return duration_minutes

def split_audio(file_path, chunk_length_ms=300000):  # 5 minutes = 300,000 ms
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    audio = AudioSegment.from_mp3(file_path)
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_filename = os.path.join('data/audio/chunks', f"{os.path.basename(file_path)[:-4]}_chunk{i}.mp3")
        chunk.export(chunk_filename, format="mp3")
        chunk_files.append(chunk_filename)
    return chunk_files

def clean_title(title):
    # Remove brackets and anything inside them
    title = re.sub(r'\[.*?\]', '', title)
    # Remove anything after and including the pipe symbol
    title = re.split(r'\|', title)[0]
    # Replace spaces and hyphens with underscores
    title = title.replace(' ', '_').replace('-', '_')
    # Remove any non-alphanumeric characters except underscores
    title = re.sub(r'[^a-zA-Z0-9_]', '', title)
    # Remove trailing underscores
    title = title.rstrip('_')
    return title.lower()

def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    transcript = result['text']
    return transcript

def save_chunk_transcript(transcript, chunk_filename):
    chunk_transcript_path = os.path.join('data/transcripts/chunks', f"{os.path.basename(chunk_filename)[:-4]}.txt")
    with open(chunk_transcript_path, "w") as f:
        f.write(transcript)
    return chunk_transcript_path

def wrap_text(text, width=80):
    wrapped_lines = []
    for paragraph in text.split('\n'):
        wrapped_lines.extend(textwrap.fill(paragraph, width=width).split('\n'))
    return '\n'.join(wrapped_lines)

def transcribe_chunks(chunk_files, title):
    full_transcript = ""
    for i, chunk_file in enumerate(tqdm(sorted(chunk_files, key=lambda x: int(re.search(r'_chunk(\d+)', x).group(1))), desc=f"Transcribing {title}")):
        chunk_transcript_path = os.path.join('data/transcripts/chunks', f"{os.path.basename(chunk_file)[:-4]}.txt")
        if os.path.exists(chunk_transcript_path):
            print(f"Transcript for {chunk_file} already exists. Skipping transcription...")
            with open(chunk_transcript_path, "r") as f:
                transcript = f.read()
        else:
            transcript = transcribe_audio(chunk_file)
            save_chunk_transcript(transcript, chunk_file)
        
        full_transcript += wrap_text(transcript) + f"\n\n({(i+1) * 5} minutes)\n\n"
    return full_transcript

def process_documentary(url, title):
    # Clean the documentary title to create a safe filename
    formatted_title = clean_title(title)
    filename = os.path.join('data/audio/full', f"{formatted_title}.mp3")
    transcript_filename = os.path.join('data/transcripts/full', f"{formatted_title}.txt")

    # Skip if transcript already exists
    if os.path.exists(transcript_filename):
        print(f"Transcript for {formatted_title} already exists. Skipping...")
        return

    # Check if audio file already exists
    if os.path.exists(filename):
        print(f"Audio for {formatted_title} already exists. Skipping download...")
        audio = AudioSegment.from_file(filename, format="mp3")
        duration_minutes = len(audio) / 60000
    else:
        # Download the audio and get duration
        duration_minutes = download_audio(url, filename)

    # Split the audio into chunks
    chunk_files = split_audio(filename)

    # Transcribe the chunks
    full_transcript = transcribe_chunks(chunk_files, formatted_title)

    # Save the full transcript
    with open(transcript_filename, "w") as f:
        f.write(full_transcript)

    return chunk_files

if __name__ == "__main__":
    os.makedirs('data/audio/full', exist_ok=True)
    os.makedirs('data/audio/chunks', exist_ok=True)
    os.makedirs('data/transcripts/full', exist_ok=True)
    os.makedirs('data/transcripts/chunks', exist_ok=True)

    # Read the documentaries CSV file
    csv_path = 'data/documentaries.csv'
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            documentary_url = row['youtube_url']
            documentary_title = row['title']
            process_documentary(documentary_url, documentary_title)
