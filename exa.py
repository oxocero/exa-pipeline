import sys
import os
import tempfile
import logging
import torch
import librosa
import soundfile as sf
import multiprocessing
import argparse
import warnings
import numpy as np
from scipy.signal import stft, istft
import whisper
from colorama import Fore, Style
from functools import partial
import csv
import json
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import nltk
import glob
import re
from pydub import AudioSegment
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# Setup logging (from script 1)
class ColouredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: Fore.BLUE + '%(asctime)s - [DEBUG] - ' + Fore.WHITE + '%(message)s' + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + '%(asctime)s - [INFO] - ' + Fore.WHITE + '%(message)s' + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + '%(asctime)s - [WARNING] - ' + Fore.WHITE + '%(message)s' + Style.RESET_ALL,
        logging.ERROR: Fore.RED + '%(asctime)s - [ERROR] - ' + Fore.WHITE + '%(message)s' + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + '%(asctime)s - [CRITICAL] - ' + Fore.WHITE + '%(message)s' + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger():
    logger = logging.getLogger('AudioProcessingLogger')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    colored_formatter = ColouredFormatter()
    ch.setFormatter(colored_formatter)
    logger.addHandler(ch)
    return logger

logger = setup_logger()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def spectral_noise_reduction(audio, sr, noise_profile=None, noise_estimate_frames=10):
    try:
        if not np.all(np.isfinite(audio)):
            logger.warning("Input contains non-finite values. Cleaning input.")
            audio = np.nan_to_num(audio, nan=0.0, posinf=np.max(audio[np.isfinite(audio)]), neginf=np.min(audio[np.isfinite(audio)]))
        audio = librosa.util.normalize(audio)
        f, t, Zxx = stft(audio)
        if noise_profile is None:
            noise_frames = np.abs(Zxx[:, :noise_estimate_frames])
            noise_profile = np.nanmean(noise_frames, axis=1)
        mask = np.maximum(np.abs(Zxx) - noise_profile[:, np.newaxis], 0)
        mask = np.minimum(mask / (np.abs(Zxx) + 1e-10), 1.0)
        phase = np.angle(Zxx)
        Zxx_cleaned = mask * np.abs(Zxx) * np.exp(1j * phase)
        _, audio_cleaned = istft(Zxx_cleaned)
        audio_cleaned = np.nan_to_num(audio_cleaned)
        return librosa.util.normalize(audio_cleaned)
    except Exception as e:
        logger.error(f"Spectral noise reduction failed: {e}")
        return audio

def advanced_audio_cleaning(audio_path, sample_rate=16000, noise_estimate_frames=10):
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        logger.info(f"Loaded audio file: {audio_path}, Sample Rate: {sr}")
        return spectral_noise_reduction(audio, sr, noise_estimate_frames=noise_estimate_frames), sr
    except Exception as e:
        logger.error(f"Error during audio cleaning: {e}")
        return None, None

# Enhanced chunk_transcription function with detailed logging and output format options
def chunk_transcription(audio_path, whisper_model, device, chunk_length=300, overlap=30):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        logger.info(f"[{device.type}] Starting chunk transcription for: {audio_path}")
        chunk_size = chunk_length * sr
        overlap_size = overlap * sr
        chunks = []
        for i, start in enumerate(range(0, len(audio), chunk_size - overlap_size)):
            chunk = audio[start:start + chunk_size]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                chunk_path = temp_file.name
                sf.write(chunk_path, chunk, sr)
            try:
                logger.info(f"[{device.type}] Processing chunk {i + 1} for {audio_path}")
                result = whisper_model.transcribe(chunk_path, fp16=(device.type == "cuda"))
                chunks.append(result['text'])
            except Exception as e:
                logger.warning(f"Error transcribing chunk {i + 1} for {audio_path}: {e}")
            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        return chunks
    except Exception as e:
        logger.error(f"Error in chunk transcription for {audio_path}: {e}")
        return None
        
def save_transcription(transcriptions, output_dir, format):
    try:
        if format == "consolidated_json":
            output_path = os.path.join(output_dir, "all_transcriptions.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcriptions, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved consolidated transcription as JSON to {output_path}")
        elif format == "consolidated_csv":
            output_path = os.path.join(output_dir, "all_transcriptions.csv")
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["File Name", "Text"])
                for entry in transcriptions:
                    writer.writerow([entry["file_name"], entry["text"]])
            logger.info(f"Saved consolidated transcription as CSV to {output_path}")
        elif format == "separate_json":
            for entry in transcriptions:
                output_path = os.path.join(output_dir, f"{entry['file_name']}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(entry, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved transcriptions in separate JSON files to {output_dir}")
        elif format == "separate_txt":
            for entry in transcriptions:
                output_path = os.path.join(output_dir, f"{entry['file_name']}.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(entry["text"])
            logger.info(f"Saved transcriptions in separate text files to {output_dir}")
        else:
            logger.error(f"Unknown output format: {format}")
    except Exception as e:
        logger.error(f"Error saving transcription: {e}")
        
def merge_audio_files(input_dir, output_file):
    wav_files = glob.glob(os.path.join(input_dir, '*.wav'))
    if not wav_files:
        raise ValueError(f"No WAV files found in directory {input_dir}")

    def chunk_sort_key(filename):
        match = re.search(r'chunk_(\d+)\.wav', filename)
        return int(match.group(1)) if match else float('inf')

    sorted_wav_files = sorted(wav_files, key=chunk_sort_key)
    merged_audio = AudioSegment.from_wav(sorted_wav_files[0])
    for wav_file in sorted_wav_files[1:]:
        audio = AudioSegment.from_wav(wav_file)
        merged_audio += audio
    merged_audio.export(output_file, format='wav')
    
def extract_epub_text(
    epub_path, 
    output_path, 
    max_pages=3, 
    skip_chapters=4, 
    chunk_size=251, 
    enable_chunking=False,
    enable_tts=False,
    speaker_wav_files=None,
    language='en'
):
    logger = setup_logger()
    logger.info(f"Starting EPUB text extraction from {epub_path}")
    
    # Initialize TTS if enabled
    tts = None
    if enable_tts:
        if not enable_chunking:
            logger.warning("TTS requires chunking to be enabled. Disabling TTS.")
            enable_tts = False
        else:
            try:
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
                logger.info("TTS model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
                enable_tts = False
    
    try:
        # Read the EPUB book
        book = epub.read_epub(epub_path)
        logger.info("EPUB file successfully loaded")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or os.getcwd(), exist_ok=True)
        
        # Prepare output file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # Track page count and chapter count
            page_count = 0
            current_chapter_index = 0
            current_chapter = "Prologue/Introduction"
            
            # Iterate through spine items (typically HTML content)
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                # Convert HTML to plain text
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                content = soup.get_text()
                
                # Detect chapter titles (basic approach)
                chapter_title = soup.find(['h1', 'h2', 'h3'])
                if chapter_title:
                    current_chapter = chapter_title.get_text().strip()
                
                # Skip chapters if needed
                if current_chapter_index < skip_chapters:
                    logger.debug(f"Skipping chapter: {current_chapter}")
                    current_chapter_index += 1
                    continue
                
                # Break if we've gone beyond the desired chapter range
                if current_chapter_index >= skip_chapters + max_pages:
                    break
                
                # Write chapter header
                logger.info(f"Processing chapter: {current_chapter}")
                outfile.write(f"\n--- {current_chapter} ---\n\n")
                
                # Prepare for TTS if enabled
                if enable_tts:
                    # Create a directory for this chapter's audio files
                    chapter_audio_dir = os.path.join(os.path.dirname(output_path), 
                                                     f"chapter_{current_chapter_index}_audio")
                    os.makedirs(chapter_audio_dir, exist_ok=True)
                    
                    # Check existing audio chunks
                    existing_chunks = sorted(glob.glob(os.path.join(chapter_audio_dir, 'chunk_*.wav')), 
                                             key=lambda x: int(re.search(r'chunk_(\d+)\.wav', x).group(1)))
                
                # Tokenize sentences
                sentences = nltk.sent_tokenize(content)
                
                # Chunk counter for TTS
                chunk_counter = 0
                
                # Collect sentences for verification
                text_chunks = []
                
                # Write sentences
                for sentence in sentences:
                    # Clean up the sentence (remove extra whitespace)
                    clean_sentence = ' '.join(sentence.split())
                    
                    # Chunking logic
                    if enable_chunking:
                        # Split sentence into chunks of specified size
                        for i in range(0, len(clean_sentence), chunk_size):
                            chunk = clean_sentence[i:i+chunk_size]
                            text_chunks.append(chunk)
                
                # If TTS is enabled, process chunks
                if enable_tts:
                    # Verify and regenerate audio if needed
                    if existing_chunks:
                        logger.info(f"Found {len(existing_chunks)} existing audio chunks for chapter {current_chapter_index}")
                        
                        # Check if number of existing chunks matches expected
                        if len(existing_chunks) != len(text_chunks):
                            logger.warning(f"Mismatch in chunk count. Regenerating audio for chapter {current_chapter_index}")
                            
                            # Clear existing audio chunks
                            for chunk_file in existing_chunks:
                                os.remove(chunk_file)
                            existing_chunks = []
                    
                    # Generate audio chunks
                    for chunk in text_chunks:
                        try:
                            # Generate unique filename for each chunk
                            chunk_filename = os.path.join(
                                chapter_audio_dir, 
                                f"chunk_{chunk_counter}.wav"
                            )
                            
                            # Perform TTS only if chunk doesn't exist
                            if not os.path.exists(chunk_filename):
                                tts.tts_to_file(
                                    text=chunk,
                                    file_path=chunk_filename,
                                    speaker_wav=speaker_wav_files or [],
                                    language=language,
                                    split_sentences=True
                                )
                            
                            chunk_counter += 1
                        except Exception as e:
                            logger.error(f"TTS generation failed for chunk: {e}")
                    
                    # Merge audio files for the chapter
                    try:
                        merged_chapter_audio = os.path.join(
                            os.path.dirname(output_path), 
                            f"chapter_{current_chapter_index}_merged.wav"
                        )
                        merge_audio_files(chapter_audio_dir, merged_chapter_audio)
                        logger.info(f"Merged audio for chapter {current_chapter_index}")
                    except Exception as e:
                        logger.error(f"Failed to merge chapter audio: {e}")
                
                # Write text chunks to output file
                for chunk in text_chunks:
                    outfile.write(f"{chunk}\n")
                
                # Estimate page count
                page_count += len(text_chunks)
                
                # Increment chapter index
                current_chapter_index += 1
                
                # Stop if we've reached page limit
                if page_count >= max_pages:
                    break
        
        # Log successful extraction
        logger.info(f"Text extracted to {output_path}. Processed {page_count} approximate pages, starting from chapter {skip_chapters+1}.")
    
    except Exception as e:
        logger.error(f"An error occurred during EPUB extraction: {e}")
        logger.error(f"Error details: {type(e).__name__} at line {e.__traceback__.tb_lineno}")


def parallel_worker(task_args):
    task_name = task_args[0]
    task_params = task_args[1]
    try:
        if task_name == "clean":
            # Handle cleaning task
            advanced_audio_cleaning(
                task_params["audio_file"],
                task_params["output_dir"],
                task_params["sample_rate"],
                task_params["noise_estimate_frames"],
                task_params["device"]
            )
        elif task_name == "transcribe":
            # Handle transcription task
            transcription = chunk_transcription(
                task_params["audio_file"],
                task_params["output_dir"],
                task_params["chunk_length"],
                task_params["overlap"],
                task_params["whisper_model_version"],
                task_params["device"]
            )
            save_transcription(transcription, task_params["output_dir"], task_params["output_format"])
        elif task_name == "epub_extract":
            # Handle epub extraction task
            extract_epub_text(
                task_params["epub_path"],
                task_params["output_path"],
                task_params["max_pages"],
                task_params["skip_chapters"],
                task_params["chunk_size"],
                task_params["enable_chunking"],
                task_params["enable_tts"],
                task_params["speaker_wav_files"],
                task_params["language"]
            )
        else:
            logger.error(f"Unknown task: {task_name}")
    except Exception as e:
        logger.error(f"Error processing task {task_name}: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(description="Audio Processing Pipeline", add_help=False)
    main_parser.add_argument("-h", "--help", action="store_true", help="Show help message")
    main_parser.add_argument("--task", choices=["clean", "transcribe", "epub_extract"], help="Task to perform")
    main_parser.add_argument("--input", help="Input audio file or directory")
    main_parser.add_argument("--output", help="Output directory")
    main_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    main_parser.add_argument("--force_cpu", action="store_true", help="Force use of CPU instead of GPU")
    main_parser.add_argument("--show_warnings", action="store_true", help="Suppress warnings if set")

    # Parse known arguments to check for --task and -h
    args, remaining = main_parser.parse_known_args()

    # Define task-specific parsers
    if args.help:
        if args.task == "clean":
            clean_parser = argparse.ArgumentParser(description="Clean audio files", prog=f"{sys.argv[0]} --task clean")
            clean_parser.add_argument("--input", required=True, help="Input audio file or directory")
            clean_parser.add_argument("--output", required=True, help="Output directory")
            clean_parser.add_argument("--sample_rate", type=int, default=44100, help="Audio file sample rate")
            clean_parser.add_argument("--noise_estimate_frames", type=int, default=10, help="Frames to estimate noise")
            clean_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
            clean_parser.add_argument("--force_cpu", action="store_true", help="Force use of CPU instead of GPU")
            clean_parser.add_argument("--show_warnings", action="store_true", help="Show warnings if set")
            clean_parser.print_help()
        elif args.task == "transcribe":
            transcribe_parser = argparse.ArgumentParser(description="Transcribe audio files", prog=f"{sys.argv[0]} --task transcribe")
            transcribe_parser.add_argument("--input", required=True, help="Input audio file or directory")
            transcribe_parser.add_argument("--output", required=True, help="Output directory")
            transcribe_parser.add_argument("--chunk_length", type=int, default=300, help="Length of audio chunks in seconds")
            transcribe_parser.add_argument("--overlap", type=int, default=30, help="Overlap between audio chunks in seconds")
            transcribe_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
            transcribe_parser.add_argument("--force_cpu", action="store_true", help="Force use of CPU instead of GPU")
            transcribe_parser.add_argument("--whisper_model_version", default="large", help="Whisper model version")
            transcribe_parser.add_argument("--output_format", choices=["consolidated_json", "consolidated_csv", "separate_json", "separate_txt"], default="consolidated_json", help="Transcription output format")
            transcribe_parser.add_argument("--show_warnings", action="store_true", help="Show warnings if set")
            transcribe_parser.print_help()
        elif args.task == "epub_extract":
            epub_extract_parser = argparse.ArgumentParser(description="Extract text from EPUB and generate TTS", prog=f"{sys.argv[0]} --task epub_extract")
            epub_extract_parser.add_argument("--epub_path", required=True, help="Path to the EPUB file")
            epub_extract_parser.add_argument("--output_path", required=True, help="Path to save the output text file")
            epub_extract_parser.add_argument("--max_pages", type=int, default=3, help="Maximum number of pages to extract")
            epub_extract_parser.add_argument("--skip_chapters", type=int, default=4, help="Number of initial chapters to skip")
            epub_extract_parser.add_argument("--chunk_size", type=int, default=251, help="Maximum length of text chunks")
            epub_extract_parser.add_argument("--enable_chunking", action="store_true", help="Enable text chunking")
            epub_extract_parser.add_argument("--enable_tts", action="store_true", help="Enable text-to-speech")
            epub_extract_parser.add_argument("--speaker_wav_files", nargs='+', help="List of WAV files for voice cloning")
            epub_extract_parser.add_argument("--language", default="en", help="Language for TTS")
            epub_extract_parser.print_help()
        else:
            main_parser.print_help()
        exit()

    # Set default values based on task
    if args.task == "transcribe":
        main_parser.add_argument("--chunk_length", type=int, default=300, help="Length of audio chunks in seconds")
        main_parser.add_argument("--overlap", type=int, default=30, help="Overlap between audio chunks in seconds")
        main_parser.add_argument("--whisper_model_version", default="large", help="Whisper model version")
        main_parser.add_argument("--output_format", choices=["consolidated_json", "consolidated_csv", "separate_json", "separate_txt"], default="consolidated_json", help="Transcription output format")
    elif args.task == "clean":
        main_parser.add_argument("--sample_rate", type=int, default=44100, help="Audio file sample rate")
        main_parser.add_argument("--noise_estimate_frames", type=int, default=10, help="Frames to estimate noise")
    elif args.task == "epub_extract":
        main_parser.add_argument("--epub_path", help="Path to the EPUB file")
        main_parser.add_argument("--output_path", help="Path to save the output text file")
        main_parser.add_argument("--max_pages", type=int, default=3, help="Maximum number of pages to extract")
        main_parser.add_argument("--skip_chapters", type=int, default=4, help="Number of initial chapters to skip")
        main_parser.add_argument("--chunk_size", type=int, default=251, help="Maximum length of text chunks")
        main_parser.add_argument("--enable_chunking", action="store_true", help="Enable text chunking")
        main_parser.add_argument("--enable_tts", action="store_true", help="Enable text-to-speech")
        main_parser.add_argument("--speaker_wav_files", nargs='+', help="List of WAV files for voice cloning")
        main_parser.add_argument("--language", default="en", help="Language for TTS")

    args = main_parser.parse_args()

    # Toggle warnings
    if args.show_warnings:
        warnings.resetwarnings()
    else:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device.type}")

    # Collect tasks based on --task argument
    tasks = []
    if args.task == "clean":
        # Collect audio files for cleaning
        if os.path.isfile(args.input):
            audio_files = [args.input]
        elif os.path.isdir(args.input):
            audio_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith((".wav", ".mp3", ".flac"))]
        else:
            logger.error("Invalid input path for cleaning")
            exit(1)
        for audio_file in audio_files:
            tasks.append(("clean", {
                "audio_file": audio_file,
                "output_dir": args.output,
                "sample_rate": args.sample_rate,
                "noise_estimate_frames": args.noise_estimate_frames,
                "device": device
            }))
    elif args.task == "transcribe":
        # Collect audio files for transcription
        if os.path.isfile(args.input):
            audio_files = [args.input]
        elif os.path.isdir(args.input):
            audio_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith((".wav", ".mp3", ".flac"))]
        else:
            logger.error("Invalid input path for transcription")
            exit(1)
        for audio_file in audio_files:
            tasks.append(("transcribe", {
                "audio_file": audio_file,
                "output_dir": args.output,
                "chunk_length": args.chunk_length,
                "overlap": args.overlap,
                "whisper_model_version": args.whisper_model_version,
                "output_format": args.output_format,
                "device": device
            }))
    elif args.task == "epub_extract":
        # Add epub_extract task
        tasks.append(("epub_extract", {
            "epub_path": args.epub_path,
            "output_path": args.output_path,
            "max_pages": args.max_pages,
            "skip_chapters": args.skip_chapters,
            "chunk_size": args.chunk_size,
            "enable_chunking": args.enable_chunking,
            "enable_tts": args.enable_tts,
            "speaker_wav_files": args.speaker_wav_files,
            "language": args.language
        }))

    # Run parallel processing
    with multiprocessing.Pool(args.workers) as pool:
        pool.map(parallel_worker, tasks)
