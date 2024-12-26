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
    """
    Perform spectral subtraction for noise reduction with improved error handling.
    
    Args:
        audio (np.ndarray): Input audio signal
        sr (int): Sample rate
        noise_profile (np.ndarray, optional): Pre-estimated noise profile
        noise_estimate_frames (int): Number of initial frames to estimate noise
    
    Returns:
        np.ndarray: Noise-reduced audio signal
    """
    try:
        # First, check and clean the input audio
        if not np.all(np.isfinite(audio)):
            logger.warning("Input contains non-finite values. Cleaning input.")
            audio = np.nan_to_num(audio, nan=0.0, posinf=np.max(audio[np.isfinite(audio)]), neginf=np.min(audio[np.isfinite(audio)]))
        
        # Normalize audio to prevent extreme values
        audio = librosa.util.normalize(audio)
        
        # Perform Short-Time Fourier Transform
        f, t, Zxx = stft(audio)
        
        # Estimate noise profile if not provided
        if noise_profile is None:
            # Take mean magnitude of first few frames as noise estimate
            noise_frames = np.abs(Zxx[:, :noise_estimate_frames])
            noise_profile = np.nanmean(noise_frames, axis=1)
        
        # Create spectral subtraction mask
        mask = np.maximum(np.abs(Zxx) - noise_profile[:, np.newaxis], 0)
        mask = np.minimum(mask / (np.abs(Zxx) + 1e-10), 1.0)
        
        # Reconstruct complex spectrum
        phase = np.angle(Zxx)
        Zxx_cleaned = mask * np.abs(Zxx) * np.exp(1j * phase)
        
        # Inverse STFT
        _, audio_cleaned = istft(Zxx_cleaned)
        
        # Final normalization and cleaning
        audio_cleaned = np.nan_to_num(audio_cleaned)
        audio_cleaned = librosa.util.normalize(audio_cleaned)
        
        return audio_cleaned
    
    except Exception as e:
        logger.error(f"Spectral noise reduction failed: {e}")
        return audio

def advanced_audio_cleaning(audio_path, sample_rate=16000, noise_estimate_frames=10):
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        logger.info(f"Loaded audio file: {audio_path}, Sample Rate: {sr}")

        audio_cleaned = spectral_noise_reduction(audio, sr, noise_estimate_frames=noise_estimate_frames)
        return audio_cleaned, sr

    except Exception as e:
        logger.error(f"Error during audio cleaning: {e}")
        return None, None

def chunk_transcription(audio_path, whisper_model, device, chunk_length=300, overlap=30):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        logger.info(f"Starting chunk transcription: {audio_path}")

        transcribed_chunks = []
        chunk_size = chunk_length * sr
        overlap_size = overlap * sr

        for start in range(0, len(audio), chunk_size - overlap_size):
            chunk = audio[start:start + chunk_size]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                chunk_path = temp_file.name
                sf.write(chunk_path, chunk, sr)

            try:
                result = whisper_model.transcribe(chunk_path, fp16=(device.type == "cuda"))
                transcribed_chunks.append(result['text'])
            except Exception as e:
                logger.warning(f"Error transcribing chunk: {e}")
            finally:
                os.remove(chunk_path)

        return " ".join(transcribed_chunks)

    except Exception as e:
        logger.error(f"Error in chunk transcription: {e}")
        return None

def process_file(args):
    audio_file, output_dir, whisper_model, device = args
    try:
        cleaned_audio, sr = advanced_audio_cleaning(audio_file)
        if cleaned_audio is None:
            raise Exception("Cleaning failed")

        transcription = chunk_transcription(audio_file, whisper_model, device)
        if not transcription:
            raise Exception("Transcription failed")

        output_file = os.path.join(output_dir, os.path.basename(audio_file) + ".txt")
        with open(output_file, 'w') as f:
            f.write(transcription)

        output_audio_file = os.path.join(output_dir, os.path.basename(audio_file))
        sf.write(output_audio_file, cleaned_audio, sr)

        return audio_file, True, None

    except Exception as e:
        return audio_file, False, str(e)

def prepare_dataset(audio_files, output_dir, whisper_model, device, max_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_args = [(audio_file, output_dir, whisper_model, device) for audio_file in audio_files]

    with multiprocessing.Pool(max_workers) as pool:
        results = list(pool.imap(process_file, process_args))

    for audio_file, success, error in results:
        if not success:
            logger.error(f"Failed for {audio_file}: {error}")

def main():
    parser = argparse.ArgumentParser(description="WIP Audio Processing Pipeline with Spectral Noise Reduction")

    # General options
    general_group = parser.add_argument_group("General options")
    general_group.add_argument("--task", choices=["clean", "transcribe", "prepare"], required=True, help="Task to perform")
    general_group.add_argument("--input", required=True, help="Input audio file or directory")
    general_group.add_argument("--output", required=True, help="Output directory")

    # Audio processing options
    audio_group = parser.add_argument_group("Audio processing options")
    audio_group.add_argument("--chunk_length", type=int, default=300, help="Length of audio chunks in seconds")
    audio_group.add_argument("--overlap", type=int, default=30, help="Overlap between audio chunks in seconds")
    audio_group.add_argument("--sample_rate", type=int, default=44100, help="Audio file sample rate")
    audio_group.add_argument("--noise_estimate_frames", type=int, default=10, help="Number of frames to estimate noise profile")

    # System settings
    system_group = parser.add_argument_group("System settings")
    system_group.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    system_group.add_argument("--force_cpu", action="store_true", help="Force use of CPU instead of GPU")
    system_group.add_argument("--whisper_model_version", default="large", help="Version of the Whisper model to use (e.g., 'base', 'small', 'medium', 'large')")

    args = parser.parse_args()
    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device.type}")

    # Corrected line: Use the proper attribute name for the model version
    whisper_model = whisper.load_model(args.whisper_model_version, device=device)

    if os.path.isfile(args.input):
        audio_files = [args.input]
    elif os.path.isdir(args.input):
        audio_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith((".wav", ".mp3", ".flac"))]
    else:
        logger.error("Invalid input path")
        return

    if args.task == "clean":
        for audio_file in audio_files:
            cleaned_audio, sr = advanced_audio_cleaning(audio_file, args.sample_rate, args.noise_estimate_frames)
            if cleaned_audio is not None:
                output_path = os.path.join(args.output, os.path.basename(audio_file))
                sf.write(output_path, cleaned_audio, sr)
                logger.info(f"Saved cleaned audio to {output_path}")

    elif args.task == "transcribe":
        for audio_file in audio_files:
            transcription = chunk_transcription(audio_file, whisper_model, device, args.chunk_length, args.overlap)
            if transcription:
                output_path = os.path.join(args.output, os.path.basename(audio_file) + ".txt")
                with open(output_path, 'w') as f:
                    f.write(transcription)
                logger.info(f"Saved transcription to {output_path}")

    elif args.task == "prepare":
        prepare_dataset(audio_files, args.output, whisper_model, device, args.workers)

if __name__ == "__main__":
    main()
