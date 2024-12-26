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
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import whisper
from colorama import Fore, Style
from functools import partial

# Setup logging
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

def chunk_transcription(audio_path, whisper_model, device, chunk_length=300, overlap=30):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        logger.info(f"Starting chunk transcription: {audio_path}")
        chunk_size = chunk_length * sr
        overlap_size = overlap * sr
        chunks = []
        for i, start in enumerate(range(0, len(audio), chunk_size - overlap_size)):
            chunk = audio[start:start + chunk_size]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                chunk_path = temp_file.name
                sf.write(chunk_path, chunk, sr)
            try:
                logger.info(f"Processing chunk {i + 1}")
                result = whisper_model.transcribe(chunk_path, fp16=(device.type == "cuda"))
                chunks.append(result['text'])
            except Exception as e:
                logger.warning(f"Error transcribing chunk {i + 1}: {e}")
            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        return " ".join(chunks)
    except Exception as e:
        logger.error(f"Error in chunk transcription: {e}")
        return None

def parallel_worker(args):
    audio_file, output_dir, task, whisper_model, device, task_args = args
    try:
        logger.info(f"Processing file: {audio_file} for task: {task}")
        if task == "clean":
            cleaned_audio, sr = advanced_audio_cleaning(
                audio_file,
                sample_rate=task_args.get("sample_rate", 44100),
                noise_estimate_frames=task_args.get("noise_estimate_frames", 10),
            )
            if cleaned_audio is not None:
                output_path = os.path.join(output_dir, os.path.basename(audio_file))
                sf.write(output_path, cleaned_audio, sr)
                logger.info(f"Saved cleaned audio to {output_path}")
            else:
                logger.error(f"Cleaning failed for {audio_file}")
        elif task == "transcribe":
            transcription = chunk_transcription(
                audio_file,
                whisper_model,
                device,
                chunk_length=task_args.get("chunk_length", 300),
                overlap=task_args.get("overlap", 30),
            )
            if transcription:
                output_path = os.path.join(output_dir, os.path.basename(audio_file) + ".txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                logger.info(f"Saved transcription to {output_path}")
            else:
                logger.error(f"Transcription failed for {audio_file}")
        else:
            logger.error(f"Unknown task: {task}")
    except Exception as e:
        logger.error(f"Error processing {audio_file} for task {task}: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Processing Pipeline")
    parser.add_argument("--task", choices=["clean", "transcribe"], required=True, help="Task to perform")
    parser.add_argument("--input", required=True, help="Input audio file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--chunk_length", type=int, default=300, help="Length of audio chunks in seconds")
    parser.add_argument("--overlap", type=int, default=30, help="Overlap between audio chunks in seconds")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Audio file sample rate")
    parser.add_argument("--noise_estimate_frames", type=int, default=10, help="Frames to estimate noise")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--force_cpu", action="store_true", help="Force use of CPU instead of GPU")
    parser.add_argument("--whisper_model_version", default="large", help="Whisper model version")
    args = parser.parse_args()

    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device.type}")

    whisper_model = whisper.load_model(args.whisper_model_version, device=device) if args.task == "transcribe" else None

    if os.path.isfile(args.input):
        audio_files = [args.input]
    elif os.path.isdir(args.input):
        audio_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith((".wav", ".mp3", ".flac"))]
    else:
        logger.error("Invalid input path")
        exit(1)

    os.makedirs(args.output, exist_ok=True)

    try:
        task_args = {
            "sample_rate": args.sample_rate,
            "noise_estimate_frames": args.noise_estimate_frames,
            "chunk_length": args.chunk_length,
            "overlap": args.overlap,
        }
        with multiprocessing.Pool(args.workers) as pool:
            pool.map(parallel_worker, [
                (audio_file, args.output, args.task, whisper_model, device, task_args)
                for audio_file in audio_files
            ])
    except Exception as e:
        logger.critical(f"Critical error in main process: {e}")
