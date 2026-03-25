"""
VN-VoiceClone Pro - Backend Server
FastAPI server t├¡ch hß╗úp VieNeu-TTS cho voice cloning tiß║┐ng Viß╗çt
"""

import os
import uuid
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ΓöÇΓöÇ Cß║Ñu h├¼nh FFmpeg cho pydub ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
try:
    import imageio_ffmpeg
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    from pydub import AudioSegment
    AudioSegment.converter = _ffmpeg_exe
    AudioSegment.ffprobe = _ffmpeg_exe
except ImportError:
    pass  # FFmpeg sß║╜ cß║ºn c├│ sß║╡n trong PATH

# ΓöÇΓöÇ Cß║Ñu h├¼nh ─æ╞░ß╗¥ng dß║½n ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TRIMMED_DIR = BASE_DIR / "trimmed"
PROFILES_DIR = BASE_DIR / "voice_profiles"

for d in [UPLOAD_DIR, OUTPUT_DIR, TRIMMED_DIR, PROFILES_DIR]:
    d.mkdir(exist_ok=True)

# ─Éoß║ín v─ân calibration tiß║┐ng Viß╗çt mß║╖c ─æß╗ïnh
CALIBRATION_TEXT = "xin ch├áo ─æ├óy l├á tr├¼nh chuyß╗ân giß╗ìng n├│i hay v├á truyß╗ün cß║úm nhß║Ñt hiß╗çn nay."

# Giß╗¢i hß║ín ref_codes ─æß╗â tr├ính tr├án context window (2048 tokens)
# Audio 5s Γëê 100 codes, 10s Γëê 200 codes. Giß╗» Γëñ120 ─æß╗â c├▓n ─æß╗º chß╗ù cho text generation.
MAX_REF_CODES = 120

# ΓöÇΓöÇ Logging ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vn-voiceclone")

# ΓöÇΓöÇ VieNeu-TTS Engine (Singleton) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
tts_engine = None


def get_tts_engine():
    """Lß║Ñy hoß║╖c khß╗ƒi tß║ío VieNeu-TTS engine (singleton)"""
    global tts_engine
    if tts_engine is None:
        logger.info("≡ƒª£ ─Éang khß╗ƒi tß║ío VieNeu-TTS engine...")
        try:
            import torch
            from vieneu import Vieneu

            if torch.cuda.is_available():
                logger.info(f"≡ƒÜÇ GPU detected: {torch.cuda.get_device_name(0)}")
                tts_engine = Vieneu(
                    backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B",  # Full model
                    backbone_device="cuda",
                    codec_device="cuda",
                )
                logger.info("Γ£à VieNeu-TTS engine ─æ├ú sß║╡n s├áng (GPU mode)!")
            else:
                logger.info("ΓÜá∩╕Å GPU kh├┤ng khß║ú dß╗Ñng, d├╣ng CPU mode")
                tts_engine = Vieneu()  # Default GGUF CPU
                logger.info("Γ£à VieNeu-TTS engine ─æ├ú sß║╡n s├áng (CPU mode)!")
        except ImportError:
            logger.error("Γ¥î Kh├┤ng t├¼m thß║Ñy th╞░ viß╗çn vieneu. H├úy c├ái: pip install vieneu")
            raise RuntimeError("VieNeu-TTS ch╞░a ─æ╞░ß╗úc c├ái ─æß║╖t")
        except Exception as e:
            logger.error(f"Γ¥î Lß╗ùi khß╗ƒi tß║ío VieNeu-TTS: {e}")
            raise RuntimeError(f"Lß╗ùi khß╗ƒi tß║ío VieNeu-TTS: {e}")
    return tts_engine


# ΓöÇΓöÇ Lifespan ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khß╗ƒi tß║ío TTS engine khi server start"""
    logger.info("≡ƒÜÇ VN-VoiceClone Pro Backend ─æang khß╗ƒi ─æß╗Öng...")
    try:
        get_tts_engine()
    except Exception as e:
        logger.warning(f"ΓÜá∩╕Å Ch╞░a thß╗â khß╗ƒi tß║ío TTS engine: {e}")
        logger.warning("   Engine sß║╜ ─æ╞░ß╗úc khß╗ƒi tß║ío khi c├│ request ─æß║ºu ti├¬n.")
    yield
    # Cleanup
    global tts_engine
    if tts_engine is not None:
        try:
            tts_engine.close()
        except Exception:
            pass
        tts_engine = None
    logger.info("≡ƒæï Server ─æ├ú dß╗½ng.")


# ΓöÇΓöÇ FastAPI App ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
app = FastAPI(
    title="VN-VoiceClone Pro API",
    description="API cho ß╗⌐ng dß╗Ñng clone giß╗ìng n├│i tiß║┐ng Viß╗çt vß╗¢i VieNeu-TTS",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS cho Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ΓöÇΓöÇ Health Check ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.get("/api/health")
async def health_check():
    """Kiß╗âm tra trß║íng th├íi server"""
    engine_ready = tts_engine is not None
    return {
        "status": "ok",
        "engine_ready": engine_ready,
        "message": "VN-VoiceClone Pro Backend ─æang hoß║ít ─æß╗Öng"
    }


# ΓöÇΓöÇ Upload Audio ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload file audio (mp3/wav).
    Trß║ú vß╗ü file_id v├á th├┤ng tin file.
    """
    # Kiß╗âm tra ─æß╗ïnh dß║íng
    allowed_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"─Éß╗ïnh dß║íng kh├┤ng hß╗ù trß╗ú: {file_ext}. Chß║Ñp nhß║¡n: {', '.join(allowed_extensions)}"
        )

    # Tß║ío ID v├á l╞░u file
    file_id = str(uuid.uuid4())[:8]
    save_filename = f"{file_id}{file_ext}"
    save_path = UPLOAD_DIR / save_filename

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Lß║Ñy th├┤ng tin audio
    try:
        data, samplerate = sf.read(str(save_path))
        duration = len(data) / samplerate
        channels = 1 if len(data.shape) == 1 else data.shape[1]
    except Exception:
        # Fallback cho MP3 bß║▒ng pydub
        try:
            from pydub import AudioSegment
            audio_seg = AudioSegment.from_file(str(save_path))
            duration = len(audio_seg) / 1000.0
            samplerate = audio_seg.frame_rate
            channels = audio_seg.channels
        except Exception as e:
            duration = 0
            samplerate = 0
            channels = 0
            logger.warning(f"Kh├┤ng thß╗â ─æß╗ìc th├┤ng tin audio: {e}")

    logger.info(f"≡ƒôü ─É├ú upload: {file.filename} ΓåÆ {save_filename} ({duration:.1f}s)")

    return {
        "file_id": file_id,
        "filename": save_filename,
        "original_name": file.filename,
        "duration": round(duration, 2),
        "sample_rate": samplerate,
        "channels": channels,
        "size": len(content),
    }


# ΓöÇΓöÇ Serve Uploaded Audio ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.get("/api/audio/uploads/{filename}")
async def serve_uploaded_audio(filename: str):
    """Phß╗Ñc vß╗Ñ file audio ─æ├ú upload"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File kh├┤ng t├¼m thß║Ñy")
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"}
    )


# ΓöÇΓöÇ Trim Audio ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.post("/api/trim")
async def trim_audio(
    filename: str = Form(...),
    start: float = Form(...),
    end: float = Form(...),
):
    """
    Cß║»t ─æoß║ín audio tß╗½ start ─æß║┐n end (gi├óy).
    Trß║ú vß╗ü ─æ╞░ß╗¥ng dß║½n file ─æ├ú cß║»t.
    """
    source_path = UPLOAD_DIR / filename
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="File nguß╗ôn kh├┤ng t├¼m thß║Ñy")

    # Validate thß╗¥i gian
    duration = end - start
    if duration < 1:
        raise HTTPException(status_code=400, detail="─Éoß║ín cß║»t phß║úi d├ái ├¡t nhß║Ñt 1 gi├óy")
    if duration > 30:
        raise HTTPException(status_code=400, detail="─Éoß║ín cß║»t kh├┤ng ─æ╞░ß╗úc qu├í 30 gi├óy")

    try:
        # D├╣ng ffmpeg trß╗▒c tiß║┐p (kh├┤ng cß║ºn ffprobe nh╞░ pydub)
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        trim_id = str(uuid.uuid4())[:8]
        trim_filename = f"trimmed_{trim_id}.wav"
        trim_path = TRIMMED_DIR / trim_filename

        # ffmpeg -i input -ss start -t duration -acodec pcm_s16le output.wav
        cmd = [
            ffmpeg_exe,
            "-y",                    # Overwrite output
            "-i", str(source_path),  # Input file
            "-ss", str(start),       # Start time
            "-t", str(duration),     # Duration
            "-acodec", "pcm_s16le",  # WAV PCM format
            "-ar", "24000",          # Sample rate (VieNeu-TTS standard)
            "-ac", "1",              # Mono
            str(trim_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr[:500]}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr[:200]}")

        logger.info(f"Γ£é∩╕Å ─É├ú cß║»t: {filename} [{start:.1f}s - {end:.1f}s] ΓåÆ {trim_filename}")

        return {
            "trimmed_filename": trim_filename,
            "duration": round(duration, 2),
            "start": start,
            "end": end,
        }

    except Exception as e:
        logger.error(f"Γ¥î Lß╗ùi cß║»t audio: {e}")
        raise HTTPException(status_code=500, detail=f"Lß╗ùi cß║»t audio: {str(e)}")


# ΓöÇΓöÇ Trim ref_codes ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
def trim_ref_codes(ref_codes, max_codes: int = MAX_REF_CODES):
    """
    Cß║»t ref_codes nß║┐u qu├í d├ái ─æß╗â tr├ính tr├án context window.
    Giß╗» phß║ºn giß╗»a (th╞░ß╗¥ng chß╗⌐a giß╗ìng r├╡ nhß║Ñt, tr├ính silence ─æß║ºu/cuß╗æi).
    """
    import torch
    if isinstance(ref_codes, (list, )):
        if len(ref_codes) <= max_codes:
            return ref_codes
        # Lß║Ñy phß║ºn giß╗»a
        start = (len(ref_codes) - max_codes) // 2
        trimmed = ref_codes[start:start + max_codes]
        logger.info(f"   Γ£é∩╕Å Trimmed ref_codes: {len(ref_codes)} ΓåÆ {len(trimmed)} codes")
        return trimmed
    # torch.Tensor hoß║╖c np.ndarray
    flat = ref_codes.flatten()
    if len(flat) <= max_codes:
        return ref_codes
    start = (len(flat) - max_codes) // 2
    trimmed = flat[start:start + max_codes]
    if isinstance(ref_codes, torch.Tensor):
        trimmed = trimmed.clone()
    logger.info(f"   Γ£é∩╕Å Trimmed ref_codes: {len(flat)} ΓåÆ {len(trimmed)} codes")
    return trimmed


# ΓöÇΓöÇ Audio Cleanup for Voice Profile ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
def clean_audio_for_profile(audio_path: Path) -> Path:
    """
    Xß╗¡ l├╜ audio nhß║╣ nh├áng tr╞░ß╗¢c khi tr├¡ch xuß║Ñt giß╗ìng:
    1. Load & resample vß╗ü 16kHz mono
    2. Cß║»t silence ─æß║ºu/cuß╗æi
    3. Lß╗ìc tß║íp ├óm NHß║╕ (giß╗» ─æß║╖c tr╞░ng giß╗ìng)
    4. High-pass filter (loß║íi tiß║┐ng ├╣ < 80Hz)
    5. Normalize volume
    """
    import librosa
    import noisereduce as nr
    from scipy.signal import butter, sosfilt

    # Load audio, resample to 16kHz mono
    wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # Cß║»t silence ─æß║ºu/cuß╗æi (top_db=30: ├¡t cß║»t h╞ín, giß╗» giß╗ìng nhß╗Å)
    wav_trimmed, _ = librosa.effects.trim(wav, top_db=30)

    # Lß╗ìc tß║íp ├óm NHß║╕ ΓÇö giß║úm prop_decrease ─æß╗â giß╗» ─æß║╖c tr╞░ng giß╗ìng
    wav_clean = nr.reduce_noise(
        y=wav_trimmed,
        sr=sr,
        stationary=True,
        prop_decrease=0.4,       # Nhß║╣ h╞ín (c┼⌐: 0.8 ΓåÆ qu├í aggressive)
        n_fft=2048,
        freq_mask_smooth_hz=500,
    )

    # High-pass filter 80Hz (loß║íi tiß║┐ng ├╣, hum)
    sos = butter(5, 80, btype='highpass', fs=sr, output='sos')
    wav_clean = sosfilt(sos, wav_clean).astype(np.float32)

    # Normalize volume (peak = 0.95)
    max_val = np.max(np.abs(wav_clean))
    if max_val > 0:
        wav_clean = wav_clean * (0.95 / max_val)

    # L╞░u file ─æ├ú xß╗¡ l├╜
    cleaned_path = TRIMMED_DIR / f"cleaned_{audio_path.name}"
    sf.write(str(cleaned_path), wav_clean, sr)

    logger.info(f"   ≡ƒôè Audio QC: {len(wav_trimmed)/sr:.1f}s ΓåÆ cleaned {len(wav_clean)/sr:.1f}s")
    return cleaned_path


# ΓöÇΓöÇ Auto Transcript (Whisper) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
_whisper_pipeline = None

def auto_transcribe(audio_path: Path) -> str:
    """
    Tß╗▒ ─æß╗Öng nhß║¡n diß╗çn v─ân bß║ún tß╗½ audio bß║▒ng Whisper.
    D├╣ng model nhß╗Å (tiny/base) ─æß╗â nhanh, chß╗ë cß║ºn ref_text ngß║»n.
    """
    global _whisper_pipeline
    try:
        if _whisper_pipeline is None:
            from transformers import pipeline
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            logger.info("≡ƒÄÖ∩╕Å Whisper model loaded for auto-transcript")

        result = _whisper_pipeline(
            str(audio_path),
            generate_kwargs={"language": "vi", "task": "transcribe"},
        )
        text = result["text"].strip()
        logger.info(f"   ≡ƒô¥ Auto-transcript: \"{text[:80]}...\"")
        return text
    except Exception as e:
        logger.warning(f"   ΓÜá∩╕Å Auto-transcript failed: {e}")
        return ""


# ΓöÇΓöÇ Voice Profile: Create ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.post("/api/create-voice-profile")
async def create_voice_profile(
    trimmed_filename: str = Form(...),
    profile_name: str = Form(...),
    ref_text: str = Form(default=""),
):
    """
    Pipeline tß║ío voice profile chß║Ñt l╞░ß╗úng cao:
    1. Lß╗ìc tß║íp ├óm + chuß║⌐n h├│a audio
    2. Auto-transcript (Whisper) nß║┐u ch╞░a c├│ ref_text
    3. Tr├¡ch xuß║Ñt ref_codes (DNA giß╗ìng n├│i)
    4. Chß║íy calibration TTS (chß╗ìn kß║┐t quß║ú tß╗æt nhß║Ñt)
    5. L╞░u profile
    """
    ref_path = TRIMMED_DIR / trimmed_filename
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail="File mß║½u giß╗ìng kh├┤ng t├¼m thß║Ñy")

    try:
        engine = get_tts_engine()
        import json as json_mod
        from datetime import datetime

        logger.info(f"≡ƒÄ¡ ─Éang tß║ío voice profile: {profile_name}")

        # B╞░ß╗¢c 0: Lß╗ìc tß║íp ├óm & chuß║⌐n h├│a audio
        cleaned_path = clean_audio_for_profile(ref_path)
        logger.info(f"   ≡ƒº╣ ─É├ú lß╗ìc tß║íp ├óm v├á chuß║⌐n h├│a audio")

        # B╞░ß╗¢c 1: Auto-transcript nß║┐u ch╞░a c├│ ref_text
        if not ref_text.strip():
            ref_text_auto = auto_transcribe(cleaned_path)
            ref_text_final = ref_text_auto if ref_text_auto else CALIBRATION_TEXT
            logger.info(f"   ≡ƒÄÖ∩╕Å Ref text (auto): \"{ref_text_final[:60]}\"")
        else:
            ref_text_final = ref_text.strip()
            logger.info(f"   ≡ƒôä Ref text (user): \"{ref_text_final[:60]}\"")

        # B╞░ß╗¢c 2: Tr├¡ch xuß║Ñt ref_codes tß╗½ audio ─æ├ú lß╗ìc
        ref_codes = engine.encode_reference(str(cleaned_path))
        ref_codes_list = ref_codes.flatten().tolist()
        logger.info(f"   ≡ƒº¼ ─É├ú tr├¡ch xuß║Ñt {len(ref_codes_list)} codes")

        # Trim ref_codes nß║┐u qu├í d├ái (tr├ính tr├án context window)
        ref_codes_trimmed = trim_ref_codes(ref_codes)

        # B╞░ß╗¢c 3: Calibration TTS ΓÇö chß║íy ─æß║┐n 5 lß║ºn, kiß╗âm tra ─æß╗ìc ─æß║ºy ─æß╗º
        best_audio = None
        best_score = -1
        best_transcript = ""
        cal_text = CALIBRATION_TEXT
        MAX_ATTEMPTS = 5

        # Temperature schedule: bß║»t ─æß║ºu ß╗òn ─æß╗ïnh, t─âng dß║ºn nß║┐u kß║┐t quß║ú k├⌐m
        temp_schedule = [0.7, 0.8, 0.9, 1.0, 0.6]
        topk_schedule = [40, 50, 50, 50, 30]

        for attempt in range(MAX_ATTEMPTS):
            temp = temp_schedule[attempt]
            topk = topk_schedule[attempt]
            try:
                audio = engine.infer(
                    text=cal_text,
                    ref_codes=ref_codes_trimmed,
                    ref_text=ref_text_final,
                    max_chars=300,   # Calibration text ngß║»n, kh├┤ng cß║ºn chia chunk
                    temperature=temp,
                    top_k=topk,
                )

                # Kiß╗âm tra ─æß╗Ö d├ái hß╗úp l├╜
                duration = len(audio) / 24000
                if duration < 1.0:
                    logger.warning(f"   ΓÜá∩╕Å Attempt {attempt+1}: qu├í ngß║»n ({duration:.1f}s), bß╗Å qua")
                    continue

                # Resample 24kHz ΓåÆ 16kHz cho Whisper
                import librosa
                audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=24000, target_sr=16000)
                tmp_cal_path = PROFILES_DIR / f"_tmp_cal_{attempt}.wav"
                sf.write(str(tmp_cal_path), audio_16k, 16000)

                # Whisper x├íc minh: audio c├│ ─æß╗ìc ─æß╗º text kh├┤ng?
                transcript = auto_transcribe(tmp_cal_path)

                # So s├ính transcript vs cal_text (word overlap)
                cal_words = set(cal_text.lower().replace(".", "").replace(",", "").split())
                trans_words = set(transcript.lower().replace(".", "").replace(",", "").split()) if transcript else set()
                overlap = len(cal_words & trans_words) / max(len(cal_words), 1)

                # Scoring: kß║┐t hß╗úp word coverage + duration hß╗úp l├╜
                expected_dur = len(cal_text) * 0.08
                dur_score = 1.0 - min(abs(duration - expected_dur) / expected_dur, 1.0)
                score = overlap * 0.7 + dur_score * 0.3

                logger.info(
                    f"   ≡ƒöä Attempt {attempt+1} (T={temp}, K={topk}): {duration:.1f}s, "
                    f"coverage={overlap:.0%}, score={score:.2f} "
                    f"| \"{transcript[:60]}\""
                )

                # X├│a file tß║ím
                tmp_cal_path.unlink(missing_ok=True)

                if score > best_score:
                    best_score = score
                    best_audio = audio
                    best_transcript = transcript

                # Nß║┐u coverage >= 80% th├¼ ─æß╗º tß╗æt, dß╗½ng
                if overlap >= 0.8:
                    logger.info(f"   Γ£à ─Éß║ít chß║Ñt l╞░ß╗úng tß╗æt, dß╗½ng calibration")
                    break

            except Exception as e:
                logger.warning(f"   ΓÜá∩╕Å Attempt {attempt+1} failed: {e}")
                continue

        if best_audio is None:
            raise RuntimeError("Kh├┤ng thß╗â tß║ío calibration audio ─æß║ít chß║Ñt l╞░ß╗úng")

        logger.info(f"   ≡ƒôï Best transcript: \"{best_transcript[:80]}\"")
        logger.info(f"   ≡ƒôè Best score: {best_score:.2f}")

        # B╞░ß╗¢c 4: L╞░u calibration audio
        profile_id = str(uuid.uuid4())[:8]
        cal_filename = f"calibration_{profile_id}.wav"
        cal_path = PROFILES_DIR / cal_filename
        sf.write(str(cal_path), best_audio, 24000)

        # B╞░ß╗¢c 5: L╞░u profile JSON (l╞░u ref_codes ─æ├ú trim)
        import torch as _torch
        if isinstance(ref_codes_trimmed, _torch.Tensor):
            trimmed_list = ref_codes_trimmed.flatten().tolist()
        elif isinstance(ref_codes_trimmed, np.ndarray):
            trimmed_list = ref_codes_trimmed.flatten().tolist()
        else:
            trimmed_list = list(ref_codes_trimmed)

        profile_data = {
            "id": profile_id,
            "name": profile_name,
            "ref_codes": trimmed_list,
            "ref_text": ref_text_final,
            "calibration_audio": cal_filename,
            "source_audio": trimmed_filename,
            "created_at": datetime.now().isoformat(),
            "quality_score": round(best_score, 3),
            "original_codes_count": len(ref_codes_list),
        }

        profile_path = PROFILES_DIR / f"{profile_id}.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json_mod.dump(profile_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Γ£à ─É├ú tß║ío voice profile: {profile_name} ({profile_id}) ΓÇö score: {best_score:.2f}")

        return {
            "id": profile_id,
            "name": profile_name,
            "calibration_audio": cal_filename,
            "codes_count": len(trimmed_list),
            "created_at": profile_data["created_at"],
        }

    except Exception as e:
        logger.error(f"Γ¥î Lß╗ùi tß║ío voice profile: {e}")
        raise HTTPException(status_code=500, detail=f"Lß╗ùi tß║ío voice profile: {str(e)}")


# ΓöÇΓöÇ Voice Profile: Create with F5-TTS ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.post("/api/create-voice-profile-f5")
async def create_voice_profile_f5(
    trimmed_filename: str = Form(...),
    profile_name: str = Form(...),
    ref_text: str = Form(default=""),
):
    """
    Tß║ío voice profile sß╗¡ dß╗Ñng F5-TTS engine.
    F5-TTS kh├┤ng cß║ºn encode ref_codes ΓÇö chß╗ë cß║ºn file audio gß╗æc + ref_text.
    Calibration: thß╗¡ ─æß╗ìc 1 c├óu mß║½u ─æß╗â x├íc nhß║¡n voice cloning hoß║ít ─æß╗Öng.
    """
    ref_path = TRIMMED_DIR / trimmed_filename
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail="File mß║½u giß╗ìng kh├┤ng t├¼m thß║Ñy")

    try:
        f5engine = get_f5tts_engine()
        import json as json_mod
        from datetime import datetime

        logger.info(f"≡ƒÄ¡ ─Éang tß║ío voice profile (F5-TTS): {profile_name}")

        # B╞░ß╗¢c 0: Lß╗ìc tß║íp ├óm & chuß║⌐n h├│a audio
        cleaned_path = clean_audio_for_profile(ref_path)
        logger.info(f"   ≡ƒº╣ ─É├ú lß╗ìc tß║íp ├óm v├á chuß║⌐n h├│a audio")

        # B╞░ß╗¢c 1: Auto-transcript nß║┐u ch╞░a c├│ ref_text
        if not ref_text.strip():
            ref_text_auto = auto_transcribe(cleaned_path)
            ref_text_final = ref_text_auto if ref_text_auto else CALIBRATION_TEXT
            logger.info(f"   ≡ƒÄÖ∩╕Å Ref text (auto): \"{ref_text_final[:60]}\"")
        else:
            ref_text_final = ref_text.strip()
            logger.info(f"   ≡ƒôä Ref text (user): \"{ref_text_final[:60]}\"")

        # B╞░ß╗¢c 2: Calibration ΓÇö tß╗òng hß╗úp c├óu mß║½u vß╗¢i F5-TTS
        cal_text = CALIBRATION_TEXT
        best_audio = None
        best_score = -1
        best_transcript = ""

        profile_id = str(uuid.uuid4())[:8]

        for attempt in range(3):
            try:
                output_tmp = OUTPUT_DIR / f"_f5_cal_{profile_id}_{attempt}.wav"
                wav, sr, _ = f5engine.infer(
                    ref_file=str(cleaned_path),
                    ref_text=ref_text_final,
                    gen_text=cal_text,
                    file_wave=str(output_tmp),
                    seed=42 + attempt,
                )

                if wav is None or len(wav) == 0:
                    data, samplerate = sf.read(str(output_tmp))
                    wav = data
                    sr = samplerate

                duration = len(wav) / sr
                if duration < 1.0:
                    logger.warning(f"   ΓÜá∩╕Å F5 attempt {attempt+1}: qu├í ngß║»n ({duration:.1f}s)")
                    output_tmp.unlink(missing_ok=True)
                    continue

                # Whisper kiß╗âm tra
                import librosa
                audio_16k = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
                tmp_path = PROFILES_DIR / f"_tmp_f5cal_{attempt}.wav"
                sf.write(str(tmp_path), audio_16k, 16000)
                transcript = auto_transcribe(tmp_path)
                tmp_path.unlink(missing_ok=True)
                output_tmp.unlink(missing_ok=True)

                # Scoring
                cal_words = set(cal_text.lower().replace(".", "").replace(",", "").split())
                trans_words = set(transcript.lower().replace(".", "").replace(",", "").split()) if transcript else set()
                overlap = len(cal_words & trans_words) / max(len(cal_words), 1)

                expected_dur = len(cal_text) * 0.08
                dur_score = 1.0 - min(abs(duration - expected_dur) / expected_dur, 1.0)
                score = overlap * 0.7 + dur_score * 0.3

                logger.info(
                    f"   ≡ƒöä F5 attempt {attempt+1}: {duration:.1f}s, "
                    f"coverage={overlap:.0%}, score={score:.2f}"
                )

                if score > best_score:
                    best_score = score
                    best_audio = wav
                    best_transcript = transcript

                if overlap >= 0.7:
                    logger.info(f"   Γ£à F5 calibration ─æß║ít chß║Ñt l╞░ß╗úng tß╗æt")
                    break

            except Exception as e:
                logger.warning(f"   ΓÜá∩╕Å F5 attempt {attempt+1} failed: {e}")
                continue

        if best_audio is None:
            raise RuntimeError("Kh├┤ng thß╗â tß║ío calibration audio vß╗¢i F5-TTS")

        # B╞░ß╗¢c 3: L╞░u calibration audio
        cal_filename = f"calibration_{profile_id}.wav"
        cal_path = PROFILES_DIR / cal_filename
        sample_rate = 24000
        sf.write(str(cal_path), best_audio, sample_rate)

        # B╞░ß╗¢c 4: L╞░u profile JSON
        # F5-TTS kh├┤ng d├╣ng ref_codes ΓÇö chß╗ë cß║ºn source_audio + ref_text
        profile_data = {
            "id": profile_id,
            "name": profile_name,
            "ref_codes": [],  # F5-TTS kh├┤ng cß║ºn ref_codes
            "ref_text": ref_text_final,
            "calibration_audio": cal_filename,
            "source_audio": trimmed_filename,
            "created_at": datetime.now().isoformat(),
            "quality_score": round(best_score, 3),
            "engine": "f5-tts",
        }

        profile_path = PROFILES_DIR / f"{profile_id}.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json_mod.dump(profile_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Γ£à ─É├ú tß║ío F5-TTS profile: {profile_name} ({profile_id}) ΓÇö score: {best_score:.2f}")

        return {
            "id": profile_id,
            "name": profile_name,
            "calibration_audio": cal_filename,
            "codes_count": 0,
            "created_at": profile_data["created_at"],
            "engine": "f5-tts",
        }

    except Exception as e:
        logger.error(f"Γ¥î Lß╗ùi tß║ío F5-TTS profile: {e}")
        raise HTTPException(status_code=500, detail=f"Lß╗ùi tß║ío voice profile (F5-TTS): {str(e)}")



# ΓöÇΓöÇ Voice Profile: List ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.get("/api/voice-profiles")
async def list_voice_profiles():
    """Danh s├ích voice profiles ─æ├ú tß║ío"""
    import json as json_mod

    profiles = []
    for f in sorted(PROFILES_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json_mod.load(fp)
                profiles.append({
                    "id": data["id"],
                    "name": data["name"],
                    "calibration_audio": data.get("calibration_audio"),
                    "created_at": data.get("created_at"),
                    "codes_count": len(data.get("ref_codes", [])),
                })
        except Exception:
            continue

    return {"profiles": profiles}


# ΓöÇΓöÇ Voice Profile: Delete ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.delete("/api/voice-profiles/{profile_id}")
async def delete_voice_profile(profile_id: str):
    """X├│a voice profile"""
    profile_path = PROFILES_DIR / f"{profile_id}.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Profile kh├┤ng t├¼m thß║Ñy")

    import json as json_mod
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json_mod.load(f)

        # X├│a calibration audio
        cal_file = PROFILES_DIR / data.get("calibration_audio", "")
        if cal_file.exists():
            cal_file.unlink()

        # X├│a profile JSON
        profile_path.unlink()
        logger.info(f"≡ƒùæ∩╕Å ─É├ú x├│a voice profile: {data.get('name')} ({profile_id})")
        return {"status": "deleted", "id": profile_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lß╗ùi x├│a profile: {str(e)}")


# ΓöÇΓöÇ Serve Profile Calibration Audio ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.get("/api/audio/profiles/{filename}")
async def serve_profile_audio(filename: str):
    """Phß╗Ñc vß╗Ñ file audio calibration cß╗ºa profile"""
    file_path = PROFILES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File kh├┤ng t├¼m thß║Ñy")
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"}
    )


# ΓöÇΓöÇ Serve Trimmed Audio ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.get("/api/audio/trimmed/{filename}")
async def serve_trimmed_audio(filename: str):
    """Phß╗Ñc vß╗Ñ file audio ─æ├ú cß║»t"""
    file_path = TRIMMED_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File kh├┤ng t├¼m thß║Ñy")
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"}
    )


# ΓöÇΓöÇ Text Preprocessing ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
import re

# Regex cho t├ích c├óu tiß║┐ng Viß╗çt
_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_RE_CLAUSE_SPLIT = re.compile(r'(?<=[,;:])\s+')


def preprocess_vietnamese_text(text: str) -> str:
    """
    Tiß╗ün xß╗¡ l├╜ v─ân bß║ún tiß║┐ng Viß╗çt cho TTS:
    - Chuß║⌐n h├│a khoß║úng trß║»ng
    - Th├¬m dß║Ñu chß║Ñm cuß╗æi c├óu nß║┐u thiß║┐u
    - T├ích d├▓ng th├ánh c├óu ri├¬ng biß╗çt
    """
    # Chuß║⌐n h├│a xuß╗æng h├áng: thay \n bß║▒ng dß║Ñu chß║Ñm + space
    text = re.sub(r'\n+', '. ', text)

    # Chuß║⌐n h├│a khoß║úng trß║»ng thß╗½a
    text = re.sub(r'\s+', ' ', text).strip()

    # ─Éß║úm bß║úo c├│ dß║Ñu chß║Ñm cuß╗æi c├óu
    if text and text[-1] not in '.!?':
        text += '.'

    # Sß╗¡a lß╗ùi dß║Ñu chß║Ñm li├¬n tiß║┐p
    text = re.sub(r'\.{2,}', '.', text)

    # ─Éß║úm bß║úo c├│ khoß║úng trß║»ng sau dß║Ñu c├óu
    text = re.sub(r'([.!?,;:])([^\s\d])', r'\1 \2', text)

    return text


def split_into_sentences(text: str, max_chars: int = 150) -> list:
    """
    T├ích v─ân bß║ún th├ánh c├íc c├óu/─æoß║ín ngß║»n, ╞░u ti├¬n:
    1. T├ích theo dß║Ñu chß║Ñm c├óu (.!?)
    2. Nß║┐u c├óu vß║½n qu├í d├ái, t├ích theo dß║Ñu phß║⌐y (,;:)
    3. Nß║┐u vß║½n qu├í d├ái, t├ích theo khoß║úng trß║»ng
    
    Mß╗ùi chunk Γëñ max_chars k├╜ tß╗▒.
    """
    if not text:
        return []

    # B╞░ß╗¢c 1: T├ích theo dß║Ñu chß║Ñm c├óu
    raw_sentences = _RE_SENTENCE_SPLIT.split(text)
    
    chunks = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        
        if len(sent) <= max_chars:
            chunks.append(sent)
        else:
            # B╞░ß╗¢c 2: T├ích theo dß║Ñu phß║⌐y
            clauses = _RE_CLAUSE_SPLIT.split(sent)
            buffer = ""
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                if buffer and len(buffer) + 1 + len(clause) > max_chars:
                    chunks.append(buffer)
                    buffer = clause
                else:
                    buffer = (buffer + " " + clause) if buffer else clause
            
            # B╞░ß╗¢c 3: Nß║┐u buffer vß║½n qu├í d├ái, t├ích theo tß╗½
            if buffer:
                if len(buffer) <= max_chars:
                    chunks.append(buffer)
                else:
                    words = buffer.split()
                    current = ""
                    for word in words:
                        if current and len(current) + 1 + len(word) > max_chars:
                            chunks.append(current)
                            current = word
                        else:
                            current = (current + " " + word) if current else word
                    if current:
                        chunks.append(current)
    
    return [c.strip() for c in chunks if c.strip()]


def synthesize_chunk_with_retry(
    engine, chunk_text: str, ref_codes, ref_text: str,
    ref_audio: str = None, max_retries: int = 3,
    base_temperature: float = 0.8, top_k: int = 50,
) -> np.ndarray:
    """
    Tß╗òng hß╗úp 1 chunk v─ân bß║ún vß╗¢i c╞í chß║┐ retry.
    Nß║┐u output qu├í ngß║»n (thiß║┐u tß╗½), thß╗¡ lß║íi vß╗¢i temperature kh├íc.
    """
    # ╞»ß╗¢c t├¡nh thß╗¥i l╞░ß╗úng: tiß║┐ng Viß╗çt ~70-90ms/k├╜ tß╗▒
    expected_duration = len(chunk_text) * 0.07
    min_acceptable = expected_duration * 0.4  # Tß╗æi thiß╗âu 40% expected
    
    best_audio = None
    best_duration = 0
    
    for attempt in range(max_retries):
        # T─âng temperature dß║ºn nß║┐u retry
        temp = base_temperature + attempt * 0.1
        temp = min(temp, 1.2)
        
        try:
            infer_kwargs = {
                "text": chunk_text,
                "max_chars": 500,       # Chunk ─æ├ú ─æ╞░ß╗úc t├ích sß║╡n, kh├┤ng cß║ºn chia tiß║┐p
                "temperature": temp,
                "top_k": top_k,
                "ref_text": ref_text,
            }
            
            if ref_audio:
                infer_kwargs["ref_audio"] = ref_audio
            else:
                infer_kwargs["ref_codes"] = ref_codes
            
            audio = engine.infer(**infer_kwargs)
            duration = len(audio) / 24000
            
            logger.info(
                f"      Chunk ({len(chunk_text)}ch): {duration:.1f}s "
                f"(expectedΓëÑ{min_acceptable:.1f}s) T={temp:.1f} "
                f"| \"{chunk_text[:50]}...\""
            )
            
            # Giß╗» kß║┐t quß║ú tß╗æt nhß║Ñt
            if duration > best_duration:
                best_duration = duration
                best_audio = audio
            
            # ─Éß╗º d├ái ΓåÆ chß║Ñp nhß║¡n
            if duration >= min_acceptable:
                return audio
                
        except Exception as e:
            logger.warning(f"      ΓÜá∩╕Å Chunk retry {attempt+1} failed: {e}")
            continue
    
    # Trß║ú vß╗ü kß║┐t quß║ú tß╗æt nhß║Ñt d├╣ kh├┤ng ─æß║ít threshold
    if best_audio is not None:
        return best_audio
    
    # Fallback: trß║ú vß╗ü silence ngß║»n
    logger.error(f"      Γ¥î Kh├┤ng thß╗â tß╗òng hß╗úp chunk: \"{chunk_text[:60]}\"")
    return np.zeros(int(24000 * 0.5), dtype=np.float32)


# ΓöÇΓöÇ Synthesize Voice ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.post("/api/synthesize")
async def synthesize_voice(
    text: str = Form(...),
    trimmed_filename: str = Form(default=""),
    ref_text: str = Form(default=""),
    voice_profile_id: str = Form(default=""),
):
    """
    Tß╗òng hß╗úp giß╗ìng n├│i tß╗½ v─ân bß║ún.
    Pipeline n├óng cß║Ñp:
    1. Tiß╗ün xß╗¡ l├╜ v─ân bß║ún
    2. T├ích th├ánh c├óu (sentence-aware chunking)
    3. Tß╗òng hß╗úp tß╗½ng chunk vß╗¢i retry nß║┐u thiß║┐u tß╗½
    4. Gh├⌐p audio vß╗¢i crossfade m╞░ß╗út
    """
    # Validate
    if not text.strip():
        raise HTTPException(status_code=400, detail="Vui l├▓ng nhß║¡p v─ân bß║ún")

    try:
        engine = get_tts_engine()
        import json as json_mod
        import torch
        from vieneu_utils.core_utils import join_audio_chunks

        # Tiß╗ün xß╗¡ l├╜ v─ân bß║ún
        processed_text = preprocess_vietnamese_text(text)
        start_time = time.time()

        # ΓöÇΓöÇ Resolve voice reference ΓöÇΓöÇ
        ref_codes_resolved = None
        ref_text_final = ""
        ref_audio_path = None

        # Mode 1: D├╣ng voice profile (╞░u ti├¬n)
        if voice_profile_id.strip():
            profile_path = PROFILES_DIR / f"{voice_profile_id}.json"
            if not profile_path.exists():
                raise HTTPException(status_code=404, detail="Voice profile kh├┤ng t├¼m thß║Ñy")

            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json_mod.load(f)

            raw_codes = profile_data["ref_codes"]
            # Trim ref_codes nß║┐u qu├í d├ái
            trimmed_codes = trim_ref_codes(raw_codes)
            if isinstance(trimmed_codes, list):
                ref_codes_resolved = torch.tensor(trimmed_codes, dtype=torch.long)
            else:
                ref_codes_resolved = trimmed_codes
            ref_text_final = profile_data.get("ref_text", "")

            logger.info(f"≡ƒÄñ Tß╗òng hß╗úp (profile: {profile_data['name']}, {len(trimmed_codes)} codes)")

        # Mode 2: D├╣ng trimmed audio
        elif trimmed_filename.strip():
            ref_path = TRIMMED_DIR / trimmed_filename
            if not ref_path.exists():
                raise HTTPException(status_code=404, detail="File mß║½u giß╗ìng kh├┤ng t├¼m thß║Ñy")

            # Lß╗ìc tß║íp ├óm audio mß║½u
            cleaned_path = TRIMMED_DIR / f"cleaned_{trimmed_filename}"
            if not cleaned_path.exists():
                cleaned_path = clean_audio_for_profile(ref_path)

            # Encode v├á trim ref_codes
            raw_ref_codes = engine.encode_reference(str(cleaned_path))
            ref_codes_resolved = trim_ref_codes(raw_ref_codes)
            ref_text_final = ref_text.strip() if ref_text.strip() else processed_text[:50]
            ref_audio_path = None  # D├╣ng ref_codes thay v├¼ ref_audio

            logger.info(f"≡ƒÄñ Tß╗òng hß╗úp (audio: {trimmed_filename})")
        else:
            raise HTTPException(status_code=400, detail="Cß║ºn voice profile hoß║╖c file mß║½u giß╗ìng")

        logger.info(f"   ≡ƒô¥ V─ân bß║ún ({len(processed_text)} chars): {processed_text[:100]}...")

        # ΓöÇΓöÇ T├ích v─ân bß║ún th├ánh c├óu ΓöÇΓöÇ
        chunks = split_into_sentences(processed_text, max_chars=150)
        logger.info(f"   ≡ƒôª T├ích th├ánh {len(chunks)} chunk(s)")

        # ΓöÇΓöÇ Tß╗òng hß╗úp tß╗½ng chunk vß╗¢i retry ΓöÇΓöÇ
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"   ≡ƒöè Chunk {i+1}/{len(chunks)}: \"{chunk[:60]}...\"")
            
            chunk_audio = synthesize_chunk_with_retry(
                engine=engine,
                chunk_text=chunk,
                ref_codes=ref_codes_resolved,
                ref_text=ref_text_final,
                ref_audio=ref_audio_path,
                max_retries=3,
                base_temperature=0.8,
                top_k=50,
            )
            audio_chunks.append(chunk_audio)

        # ΓöÇΓöÇ Gh├⌐p audio chunks ΓöÇΓöÇ
        if len(audio_chunks) == 1:
            audio_output = audio_chunks[0]
        else:
            audio_output = join_audio_chunks(
                audio_chunks,
                sr=24000,
                silence_p=0.15,      # Khoß║úng nghß╗ë tß╗▒ nhi├¬n giß╗»a c├óu
                crossfade_p=0.0,
            )

        # ΓöÇΓöÇ Post-processing: normalize volume ΓöÇΓöÇ
        if isinstance(audio_output, np.ndarray) and len(audio_output) > 0:
            max_val = np.max(np.abs(audio_output))
            if max_val > 0:
                audio_output = (audio_output * (0.95 / max_val)).astype(np.float32)

        # L╞░u kß║┐t quß║ú
        output_id = str(uuid.uuid4())[:8]
        output_filename = f"output_{output_id}.wav"
        output_path = OUTPUT_DIR / output_filename

        if isinstance(audio_output, np.ndarray):
            sf.write(str(output_path), audio_output, 24000)
        else:
            engine.save(audio_output, str(output_path))

        elapsed = time.time() - start_time
        total_duration = len(audio_output) / 24000 if isinstance(audio_output, np.ndarray) else 0
        logger.info(
            f"Γ£à Tß╗òng hß╗úp th├ánh c├┤ng: {len(chunks)} chunks, "
            f"{total_duration:.1f}s audio trong {elapsed:.1f}s ΓåÆ {output_filename}"
        )

        return {
            "output_filename": output_filename,
            "processing_time": round(elapsed, 2),
            "text_length": len(processed_text),
            "chunks_count": len(chunks),
            "audio_duration": round(total_duration, 2),
        }

    except RuntimeError as e:
        logger.error(f"Γ¥î Lß╗ùi TTS engine: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Γ¥î Lß╗ùi tß╗òng hß╗úp: {e}")
        raise HTTPException(status_code=500, detail=f"Lß╗ùi tß╗òng hß╗úp giß╗ìng n├│i: {str(e)}")


# ΓöÇΓöÇ Synthesize with Preset Voice ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.post("/api/synthesize-preset")
async def synthesize_preset_voice(
    text: str = Form(...),
    voice_id: str = Form(default=""),
):
    """
    Tß╗òng hß╗úp giß╗ìng n├│i sß╗¡ dß╗Ñng giß╗ìng preset c├│ sß║╡n cß╗ºa VieNeu-TTS.
    Chß║Ñt l╞░ß╗úng cao h╞ín voice cloning v├¼ giß╗ìng ─æ├ú ─æ╞░ß╗úc tß╗æi ╞░u sß║╡n.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Vui l├▓ng nhß║¡p v─ân bß║ún")

    try:
        engine = get_tts_engine()

        processed_text = preprocess_vietnamese_text(text)
        start_time = time.time()

        # Lß║Ñy voice preset
        voice_data = engine.get_preset_voice(voice_id if voice_id.strip() else None)

        logger.info(f"≡ƒÄñ Tß╗òng hß╗úp preset voice: {voice_id or 'default'}")
        logger.info(f"   ≡ƒô¥ V─ân bß║ún: {processed_text[:100]}...")

        # T├ích c├óu
        chunks = split_into_sentences(processed_text, max_chars=150)
        logger.info(f"   ≡ƒôª T├ích th├ánh {len(chunks)} chunk(s)")

        # Tß╗òng hß╗úp tß╗½ng chunk
        from vieneu_utils.core_utils import join_audio_chunks
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"   ≡ƒöè Chunk {i+1}/{len(chunks)}: \"{chunk[:60]}...\"")
            audio = engine.infer(
                text=chunk,
                voice=voice_data,
                max_chars=500,
                temperature=0.8,
                top_k=50,
            )
            audio_chunks.append(audio)

        # Gh├⌐p chunks
        if len(audio_chunks) == 1:
            audio_output = audio_chunks[0]
        else:
            audio_output = join_audio_chunks(
                audio_chunks, sr=24000, silence_p=0.15, crossfade_p=0.0
            )

        # Normalize
        if isinstance(audio_output, np.ndarray) and len(audio_output) > 0:
            max_val = np.max(np.abs(audio_output))
            if max_val > 0:
                audio_output = (audio_output * (0.95 / max_val)).astype(np.float32)

        # L╞░u
        output_id = str(uuid.uuid4())[:8]
        output_filename = f"output_{output_id}.wav"
        output_path = OUTPUT_DIR / output_filename
        sf.write(str(output_path), audio_output, 24000)

        elapsed = time.time() - start_time
        total_duration = len(audio_output) / 24000
        logger.info(f"Γ£à Tß╗òng hß╗úp preset th├ánh c├┤ng: {total_duration:.1f}s trong {elapsed:.1f}s")

        return {
            "output_filename": output_filename,
            "processing_time": round(elapsed, 2),
            "text_length": len(processed_text),
            "voice_id": voice_id or "default",
            "audio_duration": round(total_duration, 2),
        }

    except Exception as e:
        logger.error(f"Γ¥î Lß╗ùi tß╗òng hß╗úp preset: {e}")
        raise HTTPException(status_code=500, detail=f"Lß╗ùi tß╗òng hß╗úp: {str(e)}")


# ΓöÇΓöÇ F5-TTS Engine (Singleton) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
_f5tts_engine = None


def get_f5tts_engine():
    """Lß║Ñy hoß║╖c khß╗ƒi tß║ío F5-TTS engine (singleton)"""
    global _f5tts_engine
    if _f5tts_engine is None:
        logger.info("≡ƒÜÇ ─Éang khß╗ƒi tß║ío F5-TTS engine...")
        try:
            from f5_tts.api import F5TTS
            _f5tts_engine = F5TTS()
            logger.info("Γ£à F5-TTS engine ─æ├ú sß║╡n s├áng!")
        except Exception as e:
            logger.error(f"Γ¥î Lß╗ùi khß╗ƒi tß║ío F5-TTS: {e}")
            raise RuntimeError(f"Lß╗ùi khß╗ƒi tß║ío F5-TTS: {e}")
    return _f5tts_engine


# ΓöÇΓöÇ Synthesize with F5-TTS (High Quality Voice Cloning) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.post("/api/synthesize-f5")
async def synthesize_f5tts(
    text: str = Form(...),
    trimmed_filename: str = Form(default=""),
    voice_profile_id: str = Form(default=""),
    ref_text: str = Form(default=""),
):
    """
    Tß╗òng hß╗úp giß╗ìng n├│i vß╗¢i F5-TTS ΓÇö engine chß║Ñt l╞░ß╗úng cao.
    Zero-shot voice cloning: chß╗ë cß║ºn 3-10s audio mß║½u.
    ─Éß╗ìc ─æß║ºy ─æß╗º 100% v─ân bß║ún.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Vui l├▓ng nhß║¡p v─ân bß║ún")

    try:
        f5engine = get_f5tts_engine()

        processed_text = preprocess_vietnamese_text(text)
        start_time = time.time()

        # ΓöÇΓöÇ Resolve reference audio ΓöÇΓöÇ
        ref_audio_path = None

        if voice_profile_id.strip():
            # Lß║Ñy source audio tß╗½ profile
            import json as json_mod
            profile_path = PROFILES_DIR / f"{voice_profile_id}.json"
            if not profile_path.exists():
                raise HTTPException(status_code=404, detail="Voice profile kh├┤ng t├¼m thß║Ñy")
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json_mod.load(f)
            # T├¼m file audio gß╗æc
            source_audio = profile_data.get("source_audio", "")
            if source_audio:
                cleaned = TRIMMED_DIR / f"cleaned_{source_audio}"
                if cleaned.exists():
                    ref_audio_path = str(cleaned)
                elif (TRIMMED_DIR / source_audio).exists():
                    ref_audio_path = str(TRIMMED_DIR / source_audio)
            if not ref_text.strip():
                ref_text = profile_data.get("ref_text", "")
            logger.info(f"≡ƒÄñ F5-TTS (profile: {profile_data.get('name', 'unknown')})")

        elif trimmed_filename.strip():
            ref_path = TRIMMED_DIR / trimmed_filename
            if not ref_path.exists():
                raise HTTPException(status_code=404, detail="File mß║½u giß╗ìng kh├┤ng t├¼m thß║Ñy")
            # D├╣ng cleaned version nß║┐u c├│
            cleaned_path = TRIMMED_DIR / f"cleaned_{trimmed_filename}"
            if cleaned_path.exists():
                ref_audio_path = str(cleaned_path)
            else:
                ref_audio_path = str(ref_path)
            logger.info(f"≡ƒÄñ F5-TTS (audio: {trimmed_filename})")

        else:
            raise HTTPException(status_code=400, detail="Cß║ºn voice profile hoß║╖c file mß║½u giß╗ìng")

        if not ref_audio_path:
            raise HTTPException(status_code=400, detail="Kh├┤ng t├¼m thß║Ñy file audio mß║½u")

        # Auto-transcript nß║┐u thiß║┐u ref_text
        if not ref_text.strip():
            ref_text = auto_transcribe(Path(ref_audio_path))
            if not ref_text:
                ref_text = "xin ch├áo"

        logger.info(f"   ≡ƒô¥ V─ân bß║ún ({len(processed_text)} chars): {processed_text[:100]}...")
        logger.info(f"   ≡ƒÄÖ∩╕Å Ref text: \"{ref_text[:60]}\"")

        # ΓöÇΓöÇ T├ích c├óu v├á tß╗òng hß╗úp tß╗½ng phß║ºn ΓöÇΓöÇ
        chunks = split_into_sentences(processed_text, max_chars=200)
        logger.info(f"   ≡ƒôª T├ích th├ánh {len(chunks)} chunk(s)")

        audio_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"   ≡ƒöè Chunk {i+1}/{len(chunks)}: \"{chunk[:60]}...\"")

            output_path_tmp = OUTPUT_DIR / f"_f5_tmp_{i}.wav"
            try:
                wav, sr, _ = f5engine.infer(
                    ref_file=ref_audio_path,
                    ref_text=ref_text,
                    gen_text=chunk,
                    file_wave=str(output_path_tmp),
                    seed=42,  # Reproducible
                )
                # wav is numpy array
                if wav is not None and len(wav) > 0:
                    audio_chunks.append(wav)
                    duration = len(wav) / sr
                    logger.info(f"      Γ£ô {duration:.1f}s")
                else:
                    # ─Éß╗ìc tß╗½ file
                    data, samplerate = sf.read(str(output_path_tmp))
                    audio_chunks.append(data)
                    logger.info(f"      Γ£ô {len(data)/samplerate:.1f}s (from file)")
            except Exception as e:
                logger.warning(f"      ΓÜá∩╕Å Chunk {i+1} error: {e}")
                # Retry with shorter chunk
                continue
            finally:
                if output_path_tmp.exists():
                    output_path_tmp.unlink(missing_ok=True)

        if not audio_chunks:
            raise RuntimeError("Kh├┤ng thß╗â tß╗òng hß╗úp bß║Ñt kß╗│ chunk n├áo")

        # ΓöÇΓöÇ Gh├⌐p audio ΓöÇΓöÇ
        if len(audio_chunks) == 1:
            audio_output = audio_chunks[0]
            sample_rate = 24000  # F5-TTS default
        else:
            from vieneu_utils.core_utils import join_audio_chunks
            # Resample all chunks to same rate if needed
            sample_rate = 24000
            audio_output = join_audio_chunks(
                audio_chunks, sr=sample_rate,
                silence_p=0.15, crossfade_p=0.0
            )

        # Normalize
        if isinstance(audio_output, np.ndarray) and len(audio_output) > 0:
            max_val = np.max(np.abs(audio_output))
            if max_val > 0:
                audio_output = (audio_output * (0.95 / max_val)).astype(np.float32)

        # L╞░u
        output_id = str(uuid.uuid4())[:8]
        output_filename = f"output_{output_id}.wav"
        output_path = OUTPUT_DIR / output_filename
        sf.write(str(output_path), audio_output, sample_rate)

        elapsed = time.time() - start_time
        total_duration = len(audio_output) / sample_rate
        logger.info(
            f"Γ£à F5-TTS th├ánh c├┤ng: {len(chunks)} chunks, "
            f"{total_duration:.1f}s audio trong {elapsed:.1f}s ΓåÆ {output_filename}"
        )

        return {
            "output_filename": output_filename,
            "processing_time": round(elapsed, 2),
            "text_length": len(processed_text),
            "chunks_count": len(chunks),
            "audio_duration": round(total_duration, 2),
            "engine": "f5-tts",
        }

    except RuntimeError as e:
        logger.error(f"Γ¥î Lß╗ùi F5-TTS engine: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Γ¥î Lß╗ùi F5-TTS: {e}")
        raise HTTPException(status_code=500, detail=f"Lß╗ùi F5-TTS: {str(e)}")

# ΓöÇΓöÇ Serve Output Audio ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.get("/api/audio/outputs/{filename}")
async def serve_output_audio(filename: str):
    """Phß╗Ñc vß╗Ñ file audio ─æ├ú tß╗òng hß╗úp"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File kh├┤ng t├¼m thß║Ñy")
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=filename,
        headers={"Accept-Ranges": "bytes"}
    )


# ΓöÇΓöÇ Download Output ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.get("/api/download/{filename}")
async def download_output(filename: str):
    """Tß║úi xuß╗æng file audio ─æ├ú tß╗òng hß╗úp"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File kh├┤ng t├¼m thß║Ñy")
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=f"VN-VoiceClone_{filename}",
        headers={
            "Content-Disposition": f'attachment; filename="VN-VoiceClone_{filename}"'
        }
    )


# ΓöÇΓöÇ List Preset Voices ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.get("/api/voices")
async def list_voices():
    """Liß╗çt k├¬ c├íc giß╗ìng preset c├│ sß║╡n"""
    try:
        engine = get_tts_engine()
        voices = engine.list_preset_voices()
        return {
            "voices": [
                {"description": desc, "id": name}
                for desc, name in voices
            ]
        }
    except Exception as e:
        return {"voices": [], "error": str(e)}


# ΓöÇΓöÇ Run Server ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=12345,
        reload=True,
        log_level="info",
    )
