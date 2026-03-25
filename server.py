"""
VN-VoiceClone Pro - Backend Server
FastAPI server tích hợp VieNeu-TTS cho voice cloning tiếng Việt
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

# ── Cấu hình FFmpeg cho pydub ────────────────────────────────────
try:
    import imageio_ffmpeg
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    from pydub import AudioSegment
    AudioSegment.converter = _ffmpeg_exe
    AudioSegment.ffprobe = _ffmpeg_exe
except ImportError:
    pass  # FFmpeg sẽ cần có sẵn trong PATH

# ── Cấu hình đường dẫn ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TRIMMED_DIR = BASE_DIR / "trimmed"
PROFILES_DIR = BASE_DIR / "voice_profiles"

for d in [UPLOAD_DIR, OUTPUT_DIR, TRIMMED_DIR, PROFILES_DIR]:
    d.mkdir(exist_ok=True)

# Đoạn văn calibration tiếng Việt mặc định
CALIBRATION_TEXT = "xin chào đây là trình chuyển giọng nói hay và truyền cảm nhất hiện nay."

# Giới hạn ref_codes để tránh tràn context window (2048 tokens)
# Audio 5s ≈ 100 codes, 10s ≈ 200 codes. Giữ ≤120 để còn đủ chỗ cho text generation.
MAX_REF_CODES = 120

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vn-voiceclone")

# ── VieNeu-TTS Engine (Singleton) ────────────────────────────────
tts_engine = None


def get_tts_engine():
    """Lấy hoặc khởi tạo VieNeu-TTS engine (singleton)"""
    global tts_engine
    if tts_engine is None:
        logger.info("🦜 Đang khởi tạo VieNeu-TTS engine...")
        try:
            import torch
            from vieneu import Vieneu

            if torch.cuda.is_available():
                logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
                tts_engine = Vieneu(
                    backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B",  # Full model
                    backbone_device="cuda",
                    codec_device="cuda",
                )
                logger.info("✅ VieNeu-TTS engine đã sẵn sàng (GPU mode)!")
            else:
                logger.info("⚠️ GPU không khả dụng, dùng CPU mode")
                tts_engine = Vieneu()  # Default GGUF CPU
                logger.info("✅ VieNeu-TTS engine đã sẵn sàng (CPU mode)!")
        except ImportError:
            logger.error("❌ Không tìm thấy thư viện vieneu. Hãy cài: pip install vieneu")
            raise RuntimeError("VieNeu-TTS chưa được cài đặt")
        except Exception as e:
            logger.error(f"❌ Lỗi khởi tạo VieNeu-TTS: {e}")
            raise RuntimeError(f"Lỗi khởi tạo VieNeu-TTS: {e}")
    return tts_engine


# ── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo TTS engine khi server start"""
    logger.info("🚀 VN-VoiceClone Pro Backend đang khởi động...")
    try:
        get_tts_engine()
    except Exception as e:
        logger.warning(f"⚠️ Chưa thể khởi tạo TTS engine: {e}")
        logger.warning("   Engine sẽ được khởi tạo khi có request đầu tiên.")
    yield
    # Cleanup
    global tts_engine
    if tts_engine is not None:
        try:
            tts_engine.close()
        except Exception:
            pass
        tts_engine = None
    logger.info("👋 Server đã dừng.")


# ── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="VN-VoiceClone Pro API",
    description="API cho ứng dụng clone giọng nói tiếng Việt với VieNeu-TTS",
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


# ── Health Check ─────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """Kiểm tra trạng thái server"""
    engine_ready = tts_engine is not None
    return {
        "status": "ok",
        "engine_ready": engine_ready,
        "message": "VN-VoiceClone Pro Backend đang hoạt động"
    }


# ── Upload Audio ─────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload file audio (mp3/wav).
    Trả về file_id và thông tin file.
    """
    # Kiểm tra định dạng
    allowed_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ: {file_ext}. Chấp nhận: {', '.join(allowed_extensions)}"
        )

    # Tạo ID và lưu file
    file_id = str(uuid.uuid4())[:8]
    save_filename = f"{file_id}{file_ext}"
    save_path = UPLOAD_DIR / save_filename

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Lấy thông tin audio
    try:
        data, samplerate = sf.read(str(save_path))
        duration = len(data) / samplerate
        channels = 1 if len(data.shape) == 1 else data.shape[1]
    except Exception:
        # Fallback cho MP3 bằng pydub
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
            logger.warning(f"Không thể đọc thông tin audio: {e}")

    logger.info(f"📁 Đã upload: {file.filename} → {save_filename} ({duration:.1f}s)")

    return {
        "file_id": file_id,
        "filename": save_filename,
        "original_name": file.filename,
        "duration": round(duration, 2),
        "sample_rate": samplerate,
        "channels": channels,
        "size": len(content),
    }


# ── Serve Uploaded Audio ─────────────────────────────────────────
@app.get("/api/audio/uploads/{filename}")
async def serve_uploaded_audio(filename: str):
    """Phục vụ file audio đã upload"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File không tìm thấy")
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"}
    )


# ── Trim Audio ───────────────────────────────────────────────────
@app.post("/api/trim")
async def trim_audio(
    filename: str = Form(...),
    start: float = Form(...),
    end: float = Form(...),
):
    """
    Cắt đoạn audio từ start đến end (giây).
    Trả về đường dẫn file đã cắt.
    """
    source_path = UPLOAD_DIR / filename
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="File nguồn không tìm thấy")

    # Validate thời gian
    duration = end - start
    if duration < 1:
        raise HTTPException(status_code=400, detail="Đoạn cắt phải dài ít nhất 1 giây")
    if duration > 30:
        raise HTTPException(status_code=400, detail="Đoạn cắt không được quá 30 giây")

    try:
        # Dùng ffmpeg trực tiếp (không cần ffprobe như pydub)
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

        logger.info(f"✂️ Đã cắt: {filename} [{start:.1f}s - {end:.1f}s] → {trim_filename}")

        return {
            "trimmed_filename": trim_filename,
            "duration": round(duration, 2),
            "start": start,
            "end": end,
        }

    except Exception as e:
        logger.error(f"❌ Lỗi cắt audio: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi cắt audio: {str(e)}")


# ── Trim ref_codes ───────────────────────────────────────────────
def trim_ref_codes(ref_codes, max_codes: int = MAX_REF_CODES):
    """
    Cắt ref_codes nếu quá dài để tránh tràn context window.
    Giữ phần giữa (thường chứa giọng rõ nhất, tránh silence đầu/cuối).
    """
    import torch
    if isinstance(ref_codes, (list, )):
        if len(ref_codes) <= max_codes:
            return ref_codes
        # Lấy phần giữa
        start = (len(ref_codes) - max_codes) // 2
        trimmed = ref_codes[start:start + max_codes]
        logger.info(f"   ✂️ Trimmed ref_codes: {len(ref_codes)} → {len(trimmed)} codes")
        return trimmed
    # torch.Tensor hoặc np.ndarray
    flat = ref_codes.flatten()
    if len(flat) <= max_codes:
        return ref_codes
    start = (len(flat) - max_codes) // 2
    trimmed = flat[start:start + max_codes]
    if isinstance(ref_codes, torch.Tensor):
        trimmed = trimmed.clone()
    logger.info(f"   ✂️ Trimmed ref_codes: {len(flat)} → {len(trimmed)} codes")
    return trimmed


# ── Audio Cleanup for Voice Profile ──────────────────────────────
def clean_audio_for_profile(audio_path: Path) -> Path:
    """
    Xử lý audio nhẹ nhàng trước khi trích xuất giọng:
    1. Load & resample về 16kHz mono
    2. Cắt silence đầu/cuối
    3. Lọc tạp âm NHẸ (giữ đặc trưng giọng)
    4. High-pass filter (loại tiếng ù < 80Hz)
    5. Normalize volume
    """
    import librosa
    import noisereduce as nr
    from scipy.signal import butter, sosfilt

    # Load audio, resample to 16kHz mono
    wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # Cắt silence đầu/cuối (top_db=30: ít cắt hơn, giữ giọng nhỏ)
    wav_trimmed, _ = librosa.effects.trim(wav, top_db=30)

    # Lọc tạp âm NHẸ — giảm prop_decrease để giữ đặc trưng giọng
    wav_clean = nr.reduce_noise(
        y=wav_trimmed,
        sr=sr,
        stationary=True,
        prop_decrease=0.4,       # Nhẹ hơn (cũ: 0.8 → quá aggressive)
        n_fft=2048,
        freq_mask_smooth_hz=500,
    )

    # High-pass filter 80Hz (loại tiếng ù, hum)
    sos = butter(5, 80, btype='highpass', fs=sr, output='sos')
    wav_clean = sosfilt(sos, wav_clean).astype(np.float32)

    # Normalize volume (peak = 0.95)
    max_val = np.max(np.abs(wav_clean))
    if max_val > 0:
        wav_clean = wav_clean * (0.95 / max_val)

    # Lưu file đã xử lý
    cleaned_path = TRIMMED_DIR / f"cleaned_{audio_path.name}"
    sf.write(str(cleaned_path), wav_clean, sr)

    logger.info(f"   📊 Audio QC: {len(wav_trimmed)/sr:.1f}s → cleaned {len(wav_clean)/sr:.1f}s")
    return cleaned_path


# ── Auto Transcript (Whisper) ────────────────────────────────────
_whisper_pipeline = None

def auto_transcribe(audio_path: Path) -> str:
    """
    Tự động nhận diện văn bản từ audio bằng Whisper.
    Dùng model nhỏ (tiny/base) để nhanh, chỉ cần ref_text ngắn.
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
            logger.info("🎙️ Whisper model loaded for auto-transcript")

        result = _whisper_pipeline(
            str(audio_path),
            generate_kwargs={"language": "vi", "task": "transcribe"},
        )
        text = result["text"].strip()
        logger.info(f"   📝 Auto-transcript: \"{text[:80]}...\"")
        return text
    except Exception as e:
        logger.warning(f"   ⚠️ Auto-transcript failed: {e}")
        return ""


# ── Voice Profile: Create ────────────────────────────────────────
@app.post("/api/create-voice-profile")
async def create_voice_profile(
    trimmed_filename: str = Form(...),
    profile_name: str = Form(...),
    ref_text: str = Form(default=""),
):
    """
    Pipeline tạo voice profile chất lượng cao:
    1. Lọc tạp âm + chuẩn hóa audio
    2. Auto-transcript (Whisper) nếu chưa có ref_text
    3. Trích xuất ref_codes (DNA giọng nói)
    4. Chạy calibration TTS (chọn kết quả tốt nhất)
    5. Lưu profile
    """
    ref_path = TRIMMED_DIR / trimmed_filename
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail="File mẫu giọng không tìm thấy")

    try:
        engine = get_tts_engine()
        import json as json_mod
        from datetime import datetime

        logger.info(f"🎭 Đang tạo voice profile: {profile_name}")

        # Bước 0: Lọc tạp âm & chuẩn hóa audio
        cleaned_path = clean_audio_for_profile(ref_path)
        logger.info(f"   🧹 Đã lọc tạp âm và chuẩn hóa audio")

        # Bước 1: Auto-transcript nếu chưa có ref_text
        if not ref_text.strip():
            ref_text_auto = auto_transcribe(cleaned_path)
            ref_text_final = ref_text_auto if ref_text_auto else CALIBRATION_TEXT
            logger.info(f"   🎙️ Ref text (auto): \"{ref_text_final[:60]}\"")
        else:
            ref_text_final = ref_text.strip()
            logger.info(f"   📄 Ref text (user): \"{ref_text_final[:60]}\"")

        # Bước 2: Trích xuất ref_codes từ audio đã lọc
        ref_codes = engine.encode_reference(str(cleaned_path))
        ref_codes_list = ref_codes.flatten().tolist()
        logger.info(f"   🧬 Đã trích xuất {len(ref_codes_list)} codes")

        # Trim ref_codes nếu quá dài (tránh tràn context window)
        ref_codes_trimmed = trim_ref_codes(ref_codes)

        # Bước 3: Calibration TTS — chạy đến 5 lần, kiểm tra đọc đầy đủ
        best_audio = None
        best_score = -1
        best_transcript = ""
        cal_text = CALIBRATION_TEXT
        MAX_ATTEMPTS = 5

        # Temperature schedule: bắt đầu ổn định, tăng dần nếu kết quả kém
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
                    max_chars=300,   # Calibration text ngắn, không cần chia chunk
                    temperature=temp,
                    top_k=topk,
                )

                # Kiểm tra độ dài hợp lý
                duration = len(audio) / 24000
                if duration < 1.0:
                    logger.warning(f"   ⚠️ Attempt {attempt+1}: quá ngắn ({duration:.1f}s), bỏ qua")
                    continue

                # Resample 24kHz → 16kHz cho Whisper
                import librosa
                audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=24000, target_sr=16000)
                tmp_cal_path = PROFILES_DIR / f"_tmp_cal_{attempt}.wav"
                sf.write(str(tmp_cal_path), audio_16k, 16000)

                # Whisper xác minh: audio có đọc đủ text không?
                transcript = auto_transcribe(tmp_cal_path)

                # So sánh transcript vs cal_text (word overlap)
                cal_words = set(cal_text.lower().replace(".", "").replace(",", "").split())
                trans_words = set(transcript.lower().replace(".", "").replace(",", "").split()) if transcript else set()
                overlap = len(cal_words & trans_words) / max(len(cal_words), 1)

                # Scoring: kết hợp word coverage + duration hợp lý
                expected_dur = len(cal_text) * 0.08
                dur_score = 1.0 - min(abs(duration - expected_dur) / expected_dur, 1.0)
                score = overlap * 0.7 + dur_score * 0.3

                logger.info(
                    f"   🔄 Attempt {attempt+1} (T={temp}, K={topk}): {duration:.1f}s, "
                    f"coverage={overlap:.0%}, score={score:.2f} "
                    f"| \"{transcript[:60]}\""
                )

                # Xóa file tạm
                tmp_cal_path.unlink(missing_ok=True)

                if score > best_score:
                    best_score = score
                    best_audio = audio
                    best_transcript = transcript

                # Nếu coverage >= 80% thì đủ tốt, dừng
                if overlap >= 0.8:
                    logger.info(f"   ✅ Đạt chất lượng tốt, dừng calibration")
                    break

            except Exception as e:
                logger.warning(f"   ⚠️ Attempt {attempt+1} failed: {e}")
                continue

        if best_audio is None:
            raise RuntimeError("Không thể tạo calibration audio đạt chất lượng")

        logger.info(f"   📋 Best transcript: \"{best_transcript[:80]}\"")
        logger.info(f"   📊 Best score: {best_score:.2f}")

        # Bước 4: Lưu calibration audio
        profile_id = str(uuid.uuid4())[:8]
        cal_filename = f"calibration_{profile_id}.wav"
        cal_path = PROFILES_DIR / cal_filename
        sf.write(str(cal_path), best_audio, 24000)

        # Bước 5: Lưu profile JSON (lưu ref_codes đã trim)
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

        logger.info(f"✅ Đã tạo voice profile: {profile_name} ({profile_id}) — score: {best_score:.2f}")

        return {
            "id": profile_id,
            "name": profile_name,
            "calibration_audio": cal_filename,
            "codes_count": len(trimmed_list),
            "created_at": profile_data["created_at"],
        }

    except Exception as e:
        logger.error(f"❌ Lỗi tạo voice profile: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi tạo voice profile: {str(e)}")


# ── Voice Profile: Create with F5-TTS ────────────────────────────
@app.post("/api/create-voice-profile-f5")
async def create_voice_profile_f5(
    trimmed_filename: str = Form(...),
    profile_name: str = Form(...),
    ref_text: str = Form(default=""),
):
    """
    Tạo voice profile sử dụng F5-TTS engine.
    F5-TTS không cần encode ref_codes — chỉ cần file audio gốc + ref_text.
    Calibration: thử đọc 1 câu mẫu để xác nhận voice cloning hoạt động.
    """
    ref_path = TRIMMED_DIR / trimmed_filename
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail="File mẫu giọng không tìm thấy")

    try:
        f5engine = get_f5tts_engine()
        import json as json_mod
        from datetime import datetime

        logger.info(f"🎭 Đang tạo voice profile (F5-TTS): {profile_name}")

        # Bước 0: Lọc tạp âm & chuẩn hóa audio
        cleaned_path = clean_audio_for_profile(ref_path)
        logger.info(f"   🧹 Đã lọc tạp âm và chuẩn hóa audio")

        # Bước 1: Auto-transcript nếu chưa có ref_text
        if not ref_text.strip():
            ref_text_auto = auto_transcribe(cleaned_path)
            ref_text_final = ref_text_auto if ref_text_auto else CALIBRATION_TEXT
            logger.info(f"   🎙️ Ref text (auto): \"{ref_text_final[:60]}\"")
        else:
            ref_text_final = ref_text.strip()
            logger.info(f"   📄 Ref text (user): \"{ref_text_final[:60]}\"")

        # Bước 2: Calibration — tổng hợp câu mẫu với F5-TTS
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
                    logger.warning(f"   ⚠️ F5 attempt {attempt+1}: quá ngắn ({duration:.1f}s)")
                    output_tmp.unlink(missing_ok=True)
                    continue

                # Whisper kiểm tra
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
                    f"   🔄 F5 attempt {attempt+1}: {duration:.1f}s, "
                    f"coverage={overlap:.0%}, score={score:.2f}"
                )

                if score > best_score:
                    best_score = score
                    best_audio = wav
                    best_transcript = transcript

                if overlap >= 0.7:
                    logger.info(f"   ✅ F5 calibration đạt chất lượng tốt")
                    break

            except Exception as e:
                logger.warning(f"   ⚠️ F5 attempt {attempt+1} failed: {e}")
                continue

        if best_audio is None:
            raise RuntimeError("Không thể tạo calibration audio với F5-TTS")

        # Bước 3: Lưu calibration audio
        cal_filename = f"calibration_{profile_id}.wav"
        cal_path = PROFILES_DIR / cal_filename
        sample_rate = 24000
        sf.write(str(cal_path), best_audio, sample_rate)

        # Bước 4: Lưu profile JSON
        # F5-TTS không dùng ref_codes — chỉ cần source_audio + ref_text
        profile_data = {
            "id": profile_id,
            "name": profile_name,
            "ref_codes": [],  # F5-TTS không cần ref_codes
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

        logger.info(f"✅ Đã tạo F5-TTS profile: {profile_name} ({profile_id}) — score: {best_score:.2f}")

        return {
            "id": profile_id,
            "name": profile_name,
            "calibration_audio": cal_filename,
            "codes_count": 0,
            "created_at": profile_data["created_at"],
            "engine": "f5-tts",
        }

    except Exception as e:
        logger.error(f"❌ Lỗi tạo F5-TTS profile: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi tạo voice profile (F5-TTS): {str(e)}")



# ── Voice Profile: List ──────────────────────────────────────────
@app.get("/api/voice-profiles")
async def list_voice_profiles():
    """Danh sách voice profiles đã tạo"""
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


# ── Voice Profile: Delete ────────────────────────────────────────
@app.delete("/api/voice-profiles/{profile_id}")
async def delete_voice_profile(profile_id: str):
    """Xóa voice profile"""
    profile_path = PROFILES_DIR / f"{profile_id}.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Profile không tìm thấy")

    import json as json_mod
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json_mod.load(f)

        # Xóa calibration audio
        cal_file = PROFILES_DIR / data.get("calibration_audio", "")
        if cal_file.exists():
            cal_file.unlink()

        # Xóa profile JSON
        profile_path.unlink()
        logger.info(f"🗑️ Đã xóa voice profile: {data.get('name')} ({profile_id})")
        return {"status": "deleted", "id": profile_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xóa profile: {str(e)}")


# ── Serve Profile Calibration Audio ──────────────────────────────
@app.get("/api/audio/profiles/{filename}")
async def serve_profile_audio(filename: str):
    """Phục vụ file audio calibration của profile"""
    file_path = PROFILES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File không tìm thấy")
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"}
    )


# ── Serve Trimmed Audio ──────────────────────────────────────────
@app.get("/api/audio/trimmed/{filename}")
async def serve_trimmed_audio(filename: str):
    """Phục vụ file audio đã cắt"""
    file_path = TRIMMED_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File không tìm thấy")
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"}
    )


# ── Text Preprocessing ───────────────────────────────────────────
import re

# Regex cho tách câu tiếng Việt
_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_RE_CLAUSE_SPLIT = re.compile(r'(?<=[,;:])\s+')


def preprocess_vietnamese_text(text: str) -> str:
    """
    Tiền xử lý văn bản tiếng Việt cho TTS:
    - Chuẩn hóa khoảng trắng
    - Thêm dấu chấm cuối câu nếu thiếu
    - Tách dòng thành câu riêng biệt
    """
    # Chuẩn hóa xuống hàng: thay \n bằng dấu chấm + space
    text = re.sub(r'\n+', '. ', text)

    # Chuẩn hóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # Đảm bảo có dấu chấm cuối câu
    if text and text[-1] not in '.!?':
        text += '.'

    # Sửa lỗi dấu chấm liên tiếp
    text = re.sub(r'\.{2,}', '.', text)

    # Đảm bảo có khoảng trắng sau dấu câu
    text = re.sub(r'([.!?,;:])([^\s\d])', r'\1 \2', text)

    return text


def split_into_sentences(text: str, max_chars: int = 150) -> list:
    """
    Tách văn bản thành các câu/đoạn ngắn, ưu tiên:
    1. Tách theo dấu chấm câu (.!?)
    2. Nếu câu vẫn quá dài, tách theo dấu phẩy (,;:)
    3. Nếu vẫn quá dài, tách theo khoảng trắng
    
    Mỗi chunk ≤ max_chars ký tự.
    """
    if not text:
        return []

    # Bước 1: Tách theo dấu chấm câu
    raw_sentences = _RE_SENTENCE_SPLIT.split(text)
    
    chunks = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        
        if len(sent) <= max_chars:
            chunks.append(sent)
        else:
            # Bước 2: Tách theo dấu phẩy
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
            
            # Bước 3: Nếu buffer vẫn quá dài, tách theo từ
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
    Tổng hợp 1 chunk văn bản với cơ chế retry.
    Nếu output quá ngắn (thiếu từ), thử lại với temperature khác.
    """
    # Ước tính thời lượng: tiếng Việt ~70-90ms/ký tự
    expected_duration = len(chunk_text) * 0.07
    min_acceptable = expected_duration * 0.4  # Tối thiểu 40% expected
    
    best_audio = None
    best_duration = 0
    
    for attempt in range(max_retries):
        # Tăng temperature dần nếu retry
        temp = base_temperature + attempt * 0.1
        temp = min(temp, 1.2)
        
        try:
            infer_kwargs = {
                "text": chunk_text,
                "max_chars": 500,       # Chunk đã được tách sẵn, không cần chia tiếp
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
                f"(expected≥{min_acceptable:.1f}s) T={temp:.1f} "
                f"| \"{chunk_text[:50]}...\""
            )
            
            # Giữ kết quả tốt nhất
            if duration > best_duration:
                best_duration = duration
                best_audio = audio
            
            # Đủ dài → chấp nhận
            if duration >= min_acceptable:
                return audio
                
        except Exception as e:
            logger.warning(f"      ⚠️ Chunk retry {attempt+1} failed: {e}")
            continue
    
    # Trả về kết quả tốt nhất dù không đạt threshold
    if best_audio is not None:
        return best_audio
    
    # Fallback: trả về silence ngắn
    logger.error(f"      ❌ Không thể tổng hợp chunk: \"{chunk_text[:60]}\"")
    return np.zeros(int(24000 * 0.5), dtype=np.float32)


# ── Synthesize Voice ─────────────────────────────────────────────
@app.post("/api/synthesize")
async def synthesize_voice(
    text: str = Form(...),
    trimmed_filename: str = Form(default=""),
    ref_text: str = Form(default=""),
    voice_profile_id: str = Form(default=""),
):
    """
    Tổng hợp giọng nói từ văn bản.
    Pipeline nâng cấp:
    1. Tiền xử lý văn bản
    2. Tách thành câu (sentence-aware chunking)
    3. Tổng hợp từng chunk với retry nếu thiếu từ
    4. Ghép audio với crossfade mượt
    """
    # Validate
    if not text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản")

    try:
        engine = get_tts_engine()
        import json as json_mod
        import torch
        from vieneu_utils.core_utils import join_audio_chunks

        # Tiền xử lý văn bản
        processed_text = preprocess_vietnamese_text(text)
        start_time = time.time()

        # ── Resolve voice reference ──
        ref_codes_resolved = None
        ref_text_final = ""
        ref_audio_path = None

        # Mode 1: Dùng voice profile (ưu tiên)
        if voice_profile_id.strip():
            profile_path = PROFILES_DIR / f"{voice_profile_id}.json"
            if not profile_path.exists():
                raise HTTPException(status_code=404, detail="Voice profile không tìm thấy")

            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json_mod.load(f)

            raw_codes = profile_data["ref_codes"]
            # Trim ref_codes nếu quá dài
            trimmed_codes = trim_ref_codes(raw_codes)
            if isinstance(trimmed_codes, list):
                ref_codes_resolved = torch.tensor(trimmed_codes, dtype=torch.long)
            else:
                ref_codes_resolved = trimmed_codes
            ref_text_final = profile_data.get("ref_text", "")

            logger.info(f"🎤 Tổng hợp (profile: {profile_data['name']}, {len(trimmed_codes)} codes)")

        # Mode 2: Dùng trimmed audio
        elif trimmed_filename.strip():
            ref_path = TRIMMED_DIR / trimmed_filename
            if not ref_path.exists():
                raise HTTPException(status_code=404, detail="File mẫu giọng không tìm thấy")

            # Lọc tạp âm audio mẫu
            cleaned_path = TRIMMED_DIR / f"cleaned_{trimmed_filename}"
            if not cleaned_path.exists():
                cleaned_path = clean_audio_for_profile(ref_path)

            # Encode và trim ref_codes
            raw_ref_codes = engine.encode_reference(str(cleaned_path))
            ref_codes_resolved = trim_ref_codes(raw_ref_codes)
            ref_text_final = ref_text.strip() if ref_text.strip() else processed_text[:50]
            ref_audio_path = None  # Dùng ref_codes thay vì ref_audio

            logger.info(f"🎤 Tổng hợp (audio: {trimmed_filename})")
        else:
            raise HTTPException(status_code=400, detail="Cần voice profile hoặc file mẫu giọng")

        logger.info(f"   📝 Văn bản ({len(processed_text)} chars): {processed_text[:100]}...")

        # ── Tách văn bản thành câu ──
        chunks = split_into_sentences(processed_text, max_chars=150)
        logger.info(f"   📦 Tách thành {len(chunks)} chunk(s)")

        # ── Tổng hợp từng chunk với retry ──
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"   🔊 Chunk {i+1}/{len(chunks)}: \"{chunk[:60]}...\"")
            
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

        # ── Ghép audio chunks ──
        if len(audio_chunks) == 1:
            audio_output = audio_chunks[0]
        else:
            audio_output = join_audio_chunks(
                audio_chunks,
                sr=24000,
                silence_p=0.15,      # Khoảng nghỉ tự nhiên giữa câu
                crossfade_p=0.0,
            )

        # ── Post-processing: normalize volume ──
        if isinstance(audio_output, np.ndarray) and len(audio_output) > 0:
            max_val = np.max(np.abs(audio_output))
            if max_val > 0:
                audio_output = (audio_output * (0.95 / max_val)).astype(np.float32)

        # Lưu kết quả
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
            f"✅ Tổng hợp thành công: {len(chunks)} chunks, "
            f"{total_duration:.1f}s audio trong {elapsed:.1f}s → {output_filename}"
        )

        return {
            "output_filename": output_filename,
            "processing_time": round(elapsed, 2),
            "text_length": len(processed_text),
            "chunks_count": len(chunks),
            "audio_duration": round(total_duration, 2),
        }

    except RuntimeError as e:
        logger.error(f"❌ Lỗi TTS engine: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Lỗi tổng hợp: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi tổng hợp giọng nói: {str(e)}")


# ── Synthesize with Preset Voice ─────────────────────────────────
@app.post("/api/synthesize-preset")
async def synthesize_preset_voice(
    text: str = Form(...),
    voice_id: str = Form(default=""),
):
    """
    Tổng hợp giọng nói sử dụng giọng preset có sẵn của VieNeu-TTS.
    Chất lượng cao hơn voice cloning vì giọng đã được tối ưu sẵn.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản")

    try:
        engine = get_tts_engine()

        processed_text = preprocess_vietnamese_text(text)
        start_time = time.time()

        # Lấy voice preset
        voice_data = engine.get_preset_voice(voice_id if voice_id.strip() else None)

        logger.info(f"🎤 Tổng hợp preset voice: {voice_id or 'default'}")
        logger.info(f"   📝 Văn bản: {processed_text[:100]}...")

        # Tách câu
        chunks = split_into_sentences(processed_text, max_chars=150)
        logger.info(f"   📦 Tách thành {len(chunks)} chunk(s)")

        # Tổng hợp từng chunk
        from vieneu_utils.core_utils import join_audio_chunks
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"   🔊 Chunk {i+1}/{len(chunks)}: \"{chunk[:60]}...\"")
            audio = engine.infer(
                text=chunk,
                voice=voice_data,
                max_chars=500,
                temperature=0.8,
                top_k=50,
            )
            audio_chunks.append(audio)

        # Ghép chunks
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

        # Lưu
        output_id = str(uuid.uuid4())[:8]
        output_filename = f"output_{output_id}.wav"
        output_path = OUTPUT_DIR / output_filename
        sf.write(str(output_path), audio_output, 24000)

        elapsed = time.time() - start_time
        total_duration = len(audio_output) / 24000
        logger.info(f"✅ Tổng hợp preset thành công: {total_duration:.1f}s trong {elapsed:.1f}s")

        return {
            "output_filename": output_filename,
            "processing_time": round(elapsed, 2),
            "text_length": len(processed_text),
            "voice_id": voice_id or "default",
            "audio_duration": round(total_duration, 2),
        }

    except Exception as e:
        logger.error(f"❌ Lỗi tổng hợp preset: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi tổng hợp: {str(e)}")


# ── F5-TTS Engine (Singleton) ────────────────────────────────────
_f5tts_engine = None


def get_f5tts_engine():
    """Lấy hoặc khởi tạo F5-TTS engine (singleton)"""
    global _f5tts_engine
    if _f5tts_engine is None:
        logger.info("🚀 Đang khởi tạo F5-TTS engine...")
        try:
            from f5_tts.api import F5TTS
            _f5tts_engine = F5TTS()
            logger.info("✅ F5-TTS engine đã sẵn sàng!")
        except Exception as e:
            logger.error(f"❌ Lỗi khởi tạo F5-TTS: {e}")
            raise RuntimeError(f"Lỗi khởi tạo F5-TTS: {e}")
    return _f5tts_engine


# ── Synthesize with F5-TTS (High Quality Voice Cloning) ─────────
@app.post("/api/synthesize-f5")
async def synthesize_f5tts(
    text: str = Form(...),
    trimmed_filename: str = Form(default=""),
    voice_profile_id: str = Form(default=""),
    ref_text: str = Form(default=""),
):
    """
    Tổng hợp giọng nói với F5-TTS — engine chất lượng cao.
    Zero-shot voice cloning: chỉ cần 3-10s audio mẫu.
    Đọc đầy đủ 100% văn bản.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản")

    try:
        f5engine = get_f5tts_engine()

        processed_text = preprocess_vietnamese_text(text)
        start_time = time.time()

        # ── Resolve reference audio ──
        ref_audio_path = None

        if voice_profile_id.strip():
            # Lấy source audio từ profile
            import json as json_mod
            profile_path = PROFILES_DIR / f"{voice_profile_id}.json"
            if not profile_path.exists():
                raise HTTPException(status_code=404, detail="Voice profile không tìm thấy")
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json_mod.load(f)
            # Tìm file audio gốc
            source_audio = profile_data.get("source_audio", "")
            if source_audio:
                cleaned = TRIMMED_DIR / f"cleaned_{source_audio}"
                if cleaned.exists():
                    ref_audio_path = str(cleaned)
                elif (TRIMMED_DIR / source_audio).exists():
                    ref_audio_path = str(TRIMMED_DIR / source_audio)
            if not ref_text.strip():
                ref_text = profile_data.get("ref_text", "")
            logger.info(f"🎤 F5-TTS (profile: {profile_data.get('name', 'unknown')})")

        elif trimmed_filename.strip():
            ref_path = TRIMMED_DIR / trimmed_filename
            if not ref_path.exists():
                raise HTTPException(status_code=404, detail="File mẫu giọng không tìm thấy")
            # Dùng cleaned version nếu có
            cleaned_path = TRIMMED_DIR / f"cleaned_{trimmed_filename}"
            if cleaned_path.exists():
                ref_audio_path = str(cleaned_path)
            else:
                ref_audio_path = str(ref_path)
            logger.info(f"🎤 F5-TTS (audio: {trimmed_filename})")

        else:
            raise HTTPException(status_code=400, detail="Cần voice profile hoặc file mẫu giọng")

        if not ref_audio_path:
            raise HTTPException(status_code=400, detail="Không tìm thấy file audio mẫu")

        # Auto-transcript nếu thiếu ref_text
        if not ref_text.strip():
            ref_text = auto_transcribe(Path(ref_audio_path))
            if not ref_text:
                ref_text = "xin chào"

        logger.info(f"   📝 Văn bản ({len(processed_text)} chars): {processed_text[:100]}...")
        logger.info(f"   🎙️ Ref text: \"{ref_text[:60]}\"")

        # ── Tách câu và tổng hợp từng phần ──
        chunks = split_into_sentences(processed_text, max_chars=200)
        logger.info(f"   📦 Tách thành {len(chunks)} chunk(s)")

        audio_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"   🔊 Chunk {i+1}/{len(chunks)}: \"{chunk[:60]}...\"")

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
                    logger.info(f"      ✓ {duration:.1f}s")
                else:
                    # Đọc từ file
                    data, samplerate = sf.read(str(output_path_tmp))
                    audio_chunks.append(data)
                    logger.info(f"      ✓ {len(data)/samplerate:.1f}s (from file)")
            except Exception as e:
                logger.warning(f"      ⚠️ Chunk {i+1} error: {e}")
                # Retry with shorter chunk
                continue
            finally:
                if output_path_tmp.exists():
                    output_path_tmp.unlink(missing_ok=True)

        if not audio_chunks:
            raise RuntimeError("Không thể tổng hợp bất kỳ chunk nào")

        # ── Ghép audio ──
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

        # Lưu
        output_id = str(uuid.uuid4())[:8]
        output_filename = f"output_{output_id}.wav"
        output_path = OUTPUT_DIR / output_filename
        sf.write(str(output_path), audio_output, sample_rate)

        elapsed = time.time() - start_time
        total_duration = len(audio_output) / sample_rate
        logger.info(
            f"✅ F5-TTS thành công: {len(chunks)} chunks, "
            f"{total_duration:.1f}s audio trong {elapsed:.1f}s → {output_filename}"
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
        logger.error(f"❌ Lỗi F5-TTS engine: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Lỗi F5-TTS: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi F5-TTS: {str(e)}")

# ── Serve Output Audio ───────────────────────────────────────────
@app.get("/api/audio/outputs/{filename}")
async def serve_output_audio(filename: str):
    """Phục vụ file audio đã tổng hợp"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File không tìm thấy")
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=filename,
        headers={"Accept-Ranges": "bytes"}
    )


# ── Download Output ──────────────────────────────────────────────
@app.get("/api/download/{filename}")
async def download_output(filename: str):
    """Tải xuống file audio đã tổng hợp"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File không tìm thấy")
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=f"VN-VoiceClone_{filename}",
        headers={
            "Content-Disposition": f'attachment; filename="VN-VoiceClone_{filename}"'
        }
    )


# ── List Preset Voices ───────────────────────────────────────────
@app.get("/api/voices")
async def list_voices():
    """Liệt kê các giọng preset có sẵn"""
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


# ── Run Server ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=12345,
        reload=True,
        log_level="info",
    )
