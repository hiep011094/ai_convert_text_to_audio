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


# ── Audio Cleanup for Voice Profile ──────────────────────────────
def clean_audio_for_profile(audio_path: Path) -> Path:
    """
    Xử lý audio chuyên nghiệp trước khi trích xuất giọng:
    1. Load & resample về 16kHz mono
    2. Cắt silence đầu/cuối
    3. Lọc tạp âm (noisereduce spectral gating)
    4. High-pass filter (loại tiếng ù, hum < 80Hz)
    5. Normalize volume
    """
    import librosa
    import noisereduce as nr
    from scipy.signal import butter, sosfilt

    # Load audio, resample to 16kHz mono
    wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # Cắt silence đầu/cuối (giữ phần giọng nói)
    wav_trimmed, _ = librosa.effects.trim(wav, top_db=25)

    # Lọc tạp âm — spectral gating
    wav_clean = nr.reduce_noise(
        y=wav_trimmed,
        sr=sr,
        stationary=True,
        prop_decrease=0.8,
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

    logger.info(f"   📊 Audio QC: {len(wav_trimmed)/sr:.1f}s, SNR ước tính tốt")
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

        # Bước 3: Calibration TTS — chạy đến 3 lần, kiểm tra đọc đầy đủ
        best_audio = None
        best_score = -1
        best_transcript = ""
        cal_text = CALIBRATION_TEXT
        MAX_ATTEMPTS = 3

        for attempt in range(MAX_ATTEMPTS):
            temp = [0.5, 0.6, 0.4][attempt]  # thử 3 mức temperature
            try:
                audio = engine.infer(
                    text=cal_text,
                    ref_codes=ref_codes,
                    ref_text=ref_text_final,
                    max_chars=200,
                    temperature=temp,
                    top_k=20,
                )

                # Kiểm tra độ dài hợp lý
                duration = len(audio) / 24000
                if duration < 0.5:
                    logger.warning(f"   ⚠️ Attempt {attempt+1}: quá ngắn ({duration:.1f}s), bỏ qua")
                    continue

                # Resample 24kHz → 16kHz cho Whisper (Whisper chỉ hỗ trợ 16kHz)
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
                    f"   🔄 Attempt {attempt+1}: {duration:.1f}s, "
                    f"coverage={overlap:.0%}, score={score:.2f} "
                    f"| \"{transcript[:60]}\""
                )

                # Xóa file tạm
                tmp_cal_path.unlink(missing_ok=True)

                if score > best_score:
                    best_score = score
                    best_audio = audio
                    best_transcript = transcript

                # Nếu coverage >= 70% thì đủ tốt, dừng
                if overlap >= 0.7:
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

        # Bước 5: Lưu profile JSON
        profile_data = {
            "id": profile_id,
            "name": profile_name,
            "ref_codes": ref_codes_list,
            "ref_text": ref_text_final,
            "calibration_audio": cal_filename,
            "source_audio": trimmed_filename,
            "created_at": datetime.now().isoformat(),
            "quality_score": round(best_score, 3),
        }

        profile_path = PROFILES_DIR / f"{profile_id}.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json_mod.dump(profile_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Đã tạo voice profile: {profile_name} ({profile_id}) — score: {best_score:.2f}")

        return {
            "id": profile_id,
            "name": profile_name,
            "calibration_audio": cal_filename,
            "codes_count": len(ref_codes_list),
            "created_at": profile_data["created_at"],
        }

    except Exception as e:
        logger.error(f"❌ Lỗi tạo voice profile: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi tạo voice profile: {str(e)}")


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
    Hỗ trợ 2 mode:
    - voice_profile_id: dùng profile đã lưu (nhanh, ổn định)
    - trimmed_filename: dùng audio thô (fallback)
    """
    # Validate
    if not text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản")

    try:
        engine = get_tts_engine()
        import json as json_mod
        import torch

        # Tiền xử lý văn bản
        processed_text = preprocess_vietnamese_text(text)

        start_time = time.time()
        infer_kwargs = {
            "text": processed_text,
            "max_chars": 80,          # Chunks ngắn → đọc chính xác hơn, ít bỏ từ
            "silence_p": 0.2,         # Khoảng nghỉ giữa câu tự nhiên
            "crossfade_p": 0.05,      # Mượt khi ghép chunks
            "temperature": 0.5,       # Thấp → rõ lời, ổn định
            "top_k": 25,              # Giới hạn → đọc chính xác
        }

        # Mode 1: Dùng voice profile (ưu tiên)
        if voice_profile_id.strip():
            profile_path = PROFILES_DIR / f"{voice_profile_id}.json"
            if not profile_path.exists():
                raise HTTPException(status_code=404, detail="Voice profile không tìm thấy")

            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json_mod.load(f)

            ref_codes = torch.tensor(profile_data["ref_codes"], dtype=torch.long)
            ref_text_final = profile_data.get("ref_text", processed_text[:50])

            logger.info(f"🎤 Tổng hợp giọng nói (profile: {profile_data['name']})...")
            logger.info(f"   📝 Văn bản: {processed_text[:100]}...")

            infer_kwargs["ref_codes"] = ref_codes
            infer_kwargs["ref_text"] = ref_text_final

        # Mode 2: Dùng trimmed audio (tự động lọc tạp âm)
        elif trimmed_filename.strip():
            ref_path = TRIMMED_DIR / trimmed_filename
            if not ref_path.exists():
                raise HTTPException(status_code=404, detail="File mẫu giọng không tìm thấy")

            # Lọc tạp âm audio mẫu trước khi sử dụng
            cleaned_path = TRIMMED_DIR / f"cleaned_{trimmed_filename}"
            if not cleaned_path.exists():
                cleaned_path = clean_audio_for_profile(ref_path)

            ref_text_final = ref_text.strip() if ref_text.strip() else processed_text[:50]

            logger.info(f"🎤 Tổng hợp giọng nói (audio: {trimmed_filename})...")
            logger.info(f"   📝 Văn bản: {processed_text[:100]}...")

            infer_kwargs["ref_audio"] = str(cleaned_path)
            infer_kwargs["ref_text"] = ref_text_final
        else:
            raise HTTPException(status_code=400, detail="Cần voice profile hoặc file mẫu giọng")

        # Gọi VieNeu-TTS
        audio_output = engine.infer(**infer_kwargs)

        # ── Post-processing: nâng cấp chất lượng output ──
        if isinstance(audio_output, np.ndarray) and len(audio_output) > 0:
            import noisereduce as nr
            # Lọc nhẹ nhiễu output (giữ giọng, bỏ artifact)
            audio_output = nr.reduce_noise(
                y=audio_output.astype(np.float32),
                sr=24000,
                stationary=True,
                prop_decrease=0.3,      # Nhẹ — chỉ lọc artifact
            )
            # Normalize volume
            max_val = np.max(np.abs(audio_output))
            if max_val > 0:
                audio_output = audio_output * (0.92 / max_val)

        # Lưu kết quả
        output_id = str(uuid.uuid4())[:8]
        output_filename = f"output_{output_id}.wav"
        output_path = OUTPUT_DIR / output_filename

        # audio_output là numpy array
        if isinstance(audio_output, np.ndarray):
            sf.write(str(output_path), audio_output, 24000)
        else:
            engine.save(audio_output, str(output_path))

        elapsed = time.time() - start_time
        logger.info(f"✅ Tổng hợp thành công trong {elapsed:.1f}s → {output_filename}")

        return {
            "output_filename": output_filename,
            "processing_time": round(elapsed, 2),
            "text_length": len(processed_text),
        }

    except RuntimeError as e:
        logger.error(f"❌ Lỗi TTS engine: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Lỗi tổng hợp: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi tổng hợp giọng nói: {str(e)}")


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
        port=8000,
        reload=True,
        log_level="info",
    )
