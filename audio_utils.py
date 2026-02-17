# ============================================
# AUDIO UTILITIES FOR STT AND TTS
# ============================================
import io
import numpy as np
import soundfile as sf
import streamlit as st
from typing import Optional, Tuple
import torch

# STT imports
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    pipeline = None

# TTS imports
try:
    from kokoro_onnx import Kokoro
    TTS_AVAILABLE = True
    TTS_LIBRARY = "kokoro-onnx"
    KPipeline = None
except ImportError:
    try:
        from kokoro import KPipeline
        TTS_AVAILABLE = True
        TTS_LIBRARY = "kokoro"
        Kokoro = None
    except ImportError:
        TTS_AVAILABLE = False
        TTS_LIBRARY = None
        Kokoro = None
        KPipeline = None


# ============================================
# STT (Speech-to-Text) Functions
# ============================================

@st.cache_resource
def load_stt_model(model_name: str = "distil-whisper/distil-large-v3", device: Optional[str] = None):
    """
    Load Whisper model for speech-to-text transcription.
    Uses caching to avoid reloading the model on every call.
    
    Args:
        model_name: Hugging Face model name (default: distil-whisper/distil-large-v3)
        device: Device to use ("cuda", "cpu", or None for auto-detection)
    
    Returns:
        Pipeline object for transcription
    """
    if not STT_AVAILABLE:
        raise ImportError("transformers library is not installed. Please install it with: pip install transformers accelerate")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load model and processor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Create pipeline (dtype is already set in model, no need to specify again)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )
        
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load STT model: {str(e)}")


def convert_audio_format(audio_bytes: bytes, target_sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Convert audio bytes to numpy array with target sample rate.
    
    Args:
        audio_bytes: Raw audio bytes
        target_sample_rate: Target sample rate (default: 16000 for Whisper)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        # Read audio from bytes
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            from scipy import signal
            num_samples = int(len(audio_data) * target_sample_rate / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)
            sample_rate = target_sample_rate
        
        # Normalize audio
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        return audio_data, sample_rate
    except Exception as e:
        raise ValueError(f"Failed to convert audio format: {str(e)}")


def transcribe_audio(audio_bytes: bytes, language: str = "de", model_name: str = "distil-whisper/distil-large-v3") -> str:
    """
    Transcribe audio bytes to text using Whisper model.
    
    Args:
        audio_bytes: Raw audio bytes from microphone or file
        language: Language code (default: "de" for German)
        model_name: Model name to use
    
    Returns:
        Transcribed text
    
    Raises:
        ImportError: If STT libraries are not available
        RuntimeError: If transcription fails
        ValueError: If audio format is invalid
    """
    if not STT_AVAILABLE:
        raise ImportError("STT functionality is not available. Please install transformers library: pip install transformers accelerate")
    
    if not audio_bytes or len(audio_bytes) == 0:
        raise ValueError("Audio bytes are empty")
    
    try:
        # Load model (cached)
        pipe = load_stt_model(model_name)
        
        # Convert audio format
        audio_array, sample_rate = convert_audio_format(audio_bytes)
        
        # Validate audio array
        if len(audio_array) == 0:
            raise ValueError("Audio array is empty after conversion")
        
        # Transcribe
        result = pipe(
            {"raw": audio_array, "sampling_rate": sample_rate},
            generate_kwargs={"language": language, "task": "transcribe"}
        )
        
        transcribed_text = result.get("text", "").strip()
        return transcribed_text if transcribed_text else ""
    except ImportError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to transcribe audio: {str(e)}")


# ============================================
# TTS (Text-to-Speech) Functions
# ============================================

@st.cache_resource
def load_tts_model(model_path: str = None, voices_path: str = None):
    """
    Load Kokoro TTS model.
    Uses caching to avoid reloading the model on every call.
    
    Args:
        model_path: Path to kokoro-v1.0.onnx file (if None, will try to download)
        voices_path: Path to voices-v1.0.bin file (if None, will try to download)
    
    Returns:
        Tuple of (TTS model object, library_type)
    """
    if not TTS_AVAILABLE:
        raise ImportError("TTS functionality is not available. Please install kokoro-onnx or kokoro library.")
    
    try:
        if TTS_LIBRARY == "kokoro-onnx":
            # kokoro-onnx requires model files
            import os
            import urllib.request
            
            # Default paths in cache directory
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "kokoro-onnx")
            os.makedirs(cache_dir, exist_ok=True)
            
            if model_path is None:
                model_path = os.path.join(cache_dir, "kokoro-v1.0.onnx")
                if not os.path.exists(model_path):
                    # Try to download model file
                    model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
                    try:
                        urllib.request.urlretrieve(model_url, model_path)
                    except Exception as e:
                        raise RuntimeError(f"Failed to download model file. Please download manually from {model_url} and place at {model_path}. Error: {str(e)}")
            
            if voices_path is None:
                voices_path = os.path.join(cache_dir, "voices-v1.0.bin")
                if not os.path.exists(voices_path):
                    # Try to download voices file
                    voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
                    try:
                        urllib.request.urlretrieve(voices_url, voices_path)
                    except Exception as e:
                        raise RuntimeError(f"Failed to download voices file. Please download manually from {voices_url} and place at {voices_path}. Error: {str(e)}")
            
            model = Kokoro(model_path, voices_path)
            return model, "kokoro-onnx"
        else:
            # Fallback to kokoro (if available)
            model = KPipeline(lang_code='a')  # 'a' for American English, 'b' for British
            return model, "kokoro"
    except Exception as e:
        raise RuntimeError(f"Failed to load TTS model: {str(e)}")


def text_to_speech(text: str, language: str = "de", speed: float = 1.0) -> bytes:
    """
    Convert text to speech audio using Kokoro model.
    
    Args:
        text: Text to convert to speech
        language: Language code (default: "de" for German)
        speed: Speech speed multiplier (default: 1.0, currently not used)
    
    Returns:
        Audio bytes in WAV format
    
    Raises:
        ImportError: If TTS libraries are not available
        RuntimeError: If TTS generation fails
        ValueError: If text is empty
    """
    if not TTS_AVAILABLE:
        raise ImportError("TTS functionality is not available. Please install kokoro-onnx library: pip install kokoro-onnx")
    
    if not text or not text.strip():
        raise ValueError("Text is empty")
    
    try:
        # Load model (cached)
        model, library_type = load_tts_model()
        
        # Generate speech
        if library_type == "kokoro-onnx":
            # kokoro-onnx API: use create() method with default voice
            # Map language codes to kokoro-onnx supported languages
            # Supported: en-us, en-gb, es, fr-fr, hi, it, pt-br, ja, zh
            # German (de) is NOT supported - fallback to English
            lang_map = {
                "en": "en-us",
                "fr": "fr-fr",
                "es": "es",
                "it": "it",
                "hi": "hi",
                "pt": "pt-br",
                "ja": "ja",
                "zh": "zh",
                "de": "en-us"  # German not supported, fallback to English
            }
            lang_code = lang_map.get(language.lower(), "en-us")
            
            # Get available voices and use first one (or default)
            try:
                voices = model.get_voices()
                voice_name = voices[0] if voices else "af_sarah"  # Default voice
            except:
                voice_name = "af_sarah"  # Default fallback
            
            audio_array, sample_rate = model.create(
                text=text,
                voice=voice_name,
                speed=speed,
                lang=lang_code
            )
        else:
            # kokoro API
            audio_array = model.generate(text)
            sample_rate = 22050  # Default for kokoro
        
        # Validate audio array
        if audio_array is None or len(audio_array) == 0:
            raise ValueError("Generated audio array is empty")
        
        # Convert to bytes
        audio_bytes_io = io.BytesIO()
        sf.write(audio_bytes_io, audio_array, sample_rate, format='WAV')
        audio_bytes_io.seek(0)
        
        return audio_bytes_io.read()
    except ImportError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to generate speech: {str(e)}")


def get_audio_bytes(audio_array: np.ndarray, sample_rate: int = 22050) -> bytes:
    """
    Convert audio numpy array to bytes for Streamlit.
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Sample rate of the audio
    
    Returns:
        Audio bytes in WAV format
    """
    try:
        audio_bytes_io = io.BytesIO()
        sf.write(audio_bytes_io, audio_array, sample_rate, format='WAV')
        audio_bytes_io.seek(0)
        return audio_bytes_io.read()
    except Exception as e:
        raise ValueError(f"Failed to convert audio array to bytes: {str(e)}")


# ============================================
# Helper Functions
# ============================================

def check_audio_support() -> dict:
    """
    Check which audio features are available.
    
    Returns:
        Dictionary with availability status for STT and TTS
    """
    return {
        "stt_available": STT_AVAILABLE,
        "tts_available": TTS_AVAILABLE,
        "cuda_available": torch.cuda.is_available() if torch else False
    }
