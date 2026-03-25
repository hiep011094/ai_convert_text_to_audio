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
    # Thêm ffmpeg vào PATH để F5-TTS và các thư viện khác tìm được
    _ffmpeg_dir = str(Path(_ffmpeg_exe).parent)
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
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
    Xử lý audio thông minh — tự động phát hiện chất lượng và áp dụng
    mức lọc phù hợp. Kể cả file âm thanh tệ cũng cho ra giọng mẫu tốt.

    Pipeline:
    1. Load & resample về 24kHz mono
    2. Đánh giá SNR → chọn mức lọc (nhẹ / trung bình / mạnh)
    3. Noise reduction đa tầng (adaptive)
    4. Band-pass filter 80Hz—8kHz (loại noise cả tần thấp + cao)
    5. De-clipping (sửa méo âm thanh)
    6. Voice Activity Detection — chỉ giữ phần có giọng nói
    7. Chọn đoạn sạch nhất 3-12s
    8. Normalize volume
    """
    import librosa
    import noisereduce as nr
    from scipy.signal import butter, sosfilt

    logger.info(f"   🔧 Đang xử lý audio: {audio_path.name}")

    # ── 1. Load audio ở 24kHz (native F5-TTS) ──
    wav, sr = librosa.load(str(audio_path), sr=24000, mono=True)
    original_duration = len(wav) / sr
    logger.info(f"   📂 Audio gốc: {original_duration:.1f}s, sr={sr}")

    # ── 2. Đánh giá chất lượng audio (SNR estimation) ──
    # Tách non-speech (noise) vs speech bằng energy threshold
    rms_full = np.sqrt(np.mean(wav ** 2))
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.01 * sr)     # 10ms hop

    # Tính RMS từng frame
    frames = librosa.util.frame(wav, frame_length=frame_length, hop_length=hop_length)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))

    # Estimate noise floor = median of lowest 20% frames
    sorted_rms = np.sort(frame_rms)
    noise_floor = np.mean(sorted_rms[:max(len(sorted_rms) // 5, 1)])
    signal_level = np.mean(sorted_rms[-max(len(sorted_rms) // 3, 1):])

    if noise_floor > 0:
        snr_estimate = 20 * np.log10(signal_level / noise_floor)
    else:
        snr_estimate = 40.0  # Very clean

    # Phân loại chất lượng
    if snr_estimate >= 25:
        quality_level = "good"
        nr_passes = 1
        prop_decrease = 0.3
    elif snr_estimate >= 15:
        quality_level = "medium"
        nr_passes = 1
        prop_decrease = 0.5
    else:
        quality_level = "bad"
        nr_passes = 2
        prop_decrease = 0.7

    logger.info(
        f"   📊 SNR ≈ {snr_estimate:.0f}dB → chất lượng: {quality_level} "
        f"(nr_passes={nr_passes}, prop={prop_decrease})"
    )

    # ── 3. Cắt silence đầu/cuối ──
    top_db = 25 if quality_level == "bad" else 28
    wav_trimmed, _ = librosa.effects.trim(wav, top_db=top_db)

    # ── 4. Noise reduction đa tầng (adaptive) ──
    wav_clean = wav_trimmed.copy()
    for nr_pass in range(nr_passes):
        wav_clean = nr.reduce_noise(
            y=wav_clean,
            sr=sr,
            stationary=True if nr_pass == 0 else False,  # Pass 2: non-stationary
            prop_decrease=prop_decrease,
            n_fft=2048,
            freq_mask_smooth_hz=500,
        ).astype(np.float32)
        if nr_pass == 0 and nr_passes > 1:
            logger.info(f"   🔇 Pass {nr_pass+1} noise reduction done")

    # ── 5. Band-pass filter 80Hz—8kHz ──
    # Loại cả tiếng ù tần thấp VÀ tiếng xì/rít tần cao
    sos_hp = butter(5, 80, btype='highpass', fs=sr, output='sos')
    sos_lp = butter(4, 8000, btype='lowpass', fs=sr, output='sos')
    wav_clean = sosfilt(sos_hp, wav_clean).astype(np.float32)
    wav_clean = sosfilt(sos_lp, wav_clean).astype(np.float32)

    # ── 6. De-clipping (sửa méo nếu audio bị clip) ──
    clip_threshold = 0.99
    clipped_ratio = np.mean(np.abs(wav_clean) > clip_threshold)
    if clipped_ratio > 0.01:  # >1% samples bị clip
        logger.info(f"   🔧 Phát hiện clipping ({clipped_ratio:.1%}), đang sửa...")
        # Soft-clip bằng tanh
        wav_clean = np.tanh(wav_clean * 0.8).astype(np.float32)

    # ── 7. Loại bỏ khoảng ngắt quãng + ghép liền mạch (crossfade) ──
    # Chạy cho TẤT CẢ audio, không chỉ "bad" — audio ngắt quãng = giọng mẫu tệ
    gap_top_db = 22 if quality_level == "bad" else 26
    speech_intervals = librosa.effects.split(wav_clean, top_db=gap_top_db, frame_length=2048, hop_length=512)

    if len(speech_intervals) > 1:
        # Có nhiều đoạn giọng → ghép liền mạch với crossfade
        crossfade_samples = int(0.015 * sr)  # 15ms crossfade
        segments = []
        for start, end in speech_intervals:
            seg = wav_clean[start:end]
            if len(seg) / sr >= 0.15:  # Chỉ giữ đoạn >= 150ms
                segments.append(seg)

        if len(segments) > 1:
            # Ghép các đoạn với crossfade
            joined = segments[0].copy()
            for seg in segments[1:]:
                if len(joined) >= crossfade_samples and len(seg) >= crossfade_samples:
                    # Crossfade overlap
                    fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
                    joined[-crossfade_samples:] *= fade_out
                    seg_copy = seg.copy()
                    seg_copy[:crossfade_samples] *= fade_in
                    joined[-crossfade_samples:] += seg_copy[:crossfade_samples]
                    joined = np.concatenate([joined, seg_copy[crossfade_samples:]])
                else:
                    joined = np.concatenate([joined, seg])
            
            gaps_removed = len(speech_intervals) - 1
            old_dur = len(wav_clean) / sr
            wav_clean = joined
            new_dur = len(wav_clean) / sr
            logger.info(
                f"   🔗 Ghép liền {len(segments)} đoạn giọng (bỏ {gaps_removed} khoảng trống): "
                f"{old_dur:.1f}s → {new_dur:.1f}s"
            )
        elif len(segments) == 1:
            wav_clean = segments[0]
    elif len(speech_intervals) == 1:
        # Chỉ 1 đoạn liên tục → cắt chính xác
        start, end = speech_intervals[0]
        wav_clean = wav_clean[start:end]

    # ── 8. Chọn đoạn tốt nhất 3-12s ──
    max_samples = 12 * sr
    min_samples = 3 * sr
    if len(wav_clean) > max_samples:
        # Tìm đoạn có năng lượng đều nhất (ổn định nhất)
        best_start = 0
        best_rms_std = float('inf')
        chunk_size = max_samples
        step = sr  # Bước nhảy 1s

        for start in range(0, len(wav_clean) - chunk_size + 1, step):
            chunk = wav_clean[start:start + chunk_size]
            c_frames = librosa.util.frame(chunk, frame_length=frame_length, hop_length=hop_length)
            c_rms = np.sqrt(np.mean(c_frames ** 2, axis=0))
            rms_std = np.std(c_rms)
            if rms_std < best_rms_std and np.mean(c_rms) > noise_floor * 2:
                best_rms_std = rms_std
                best_start = start

        wav_clean = wav_clean[best_start:best_start + chunk_size]
        logger.info(f"   ✂️ Chọn đoạn ổn định nhất: {best_start/sr:.1f}s—{(best_start+chunk_size)/sr:.1f}s")
    elif len(wav_clean) < min_samples:
        logger.warning(f"   ⚠️ Audio ngắn ({len(wav_clean)/sr:.1f}s) — nên dùng >3s")

    # ── 9. Normalize volume (peak = 0.95) ──
    max_val = np.max(np.abs(wav_clean))
    if max_val > 0:
        wav_clean = (wav_clean * (0.95 / max_val)).astype(np.float32)

    # ── 10. Lưu bản 24kHz ──
    cleaned_path = TRIMMED_DIR / f"cleaned_{audio_path.name}"
    sf.write(str(cleaned_path), wav_clean, sr)

    final_duration = len(wav_clean) / sr
    final_rms = np.sqrt(np.mean(wav_clean ** 2))
    logger.info(
        f"   ✅ Audio QC hoàn tất: {original_duration:.1f}s → {final_duration:.1f}s "
        f"(24kHz, quality={quality_level}, rms={final_rms:.3f})"
    )
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
    Calibration nâng cấp: multi-text, silence removal, Whisper verification.
    """
    ref_path = TRIMMED_DIR / trimmed_filename
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail="File mẫu giọng không tìm thấy")

    # Task ID cho progress tracking
    task_id = str(uuid.uuid4())[:8]
    _synthesis_progress[task_id] = {
        "status": "processing", "progress": 0,
        "message": "Đang khởi tạo...", "task_id": task_id,
    }

    try:
        f5engine = get_f5tts_engine()
        import json as json_mod
        from datetime import datetime
        import random as _random

        logger.info(f"🎭 Đang tạo voice profile (F5-TTS): {profile_name}")

        # Bước 1: Lọc tạp âm & chuẩn hóa audio
        _synthesis_progress[task_id].update(progress=10, message="🧹 Đang lọc tạp âm audio...")
        cleaned_path = clean_audio_for_profile(ref_path)
        logger.info(f"   🧹 Đã lọc tạp âm và chuẩn hóa audio")

        # Bước 2: Auto-transcript nếu chưa có ref_text
        _synthesis_progress[task_id].update(progress=20, message="🎙️ Đang nhận dạng giọng nói (Whisper)...")
        if not ref_text.strip():
            ref_text_auto = auto_transcribe(cleaned_path)
            ref_text_final = ref_text_auto if ref_text_auto else CALIBRATION_TEXT
            logger.info(f"   🎙️ Ref text (auto): \"{ref_text_final[:60]}\"")
        else:
            ref_text_final = ref_text.strip()
            logger.info(f"   📄 Ref text (user): \"{ref_text_final[:60]}\"")

        # Bước 3: Calibration chất lượng cao + ổn định
        _synthesis_progress[task_id].update(progress=25, message="🔊 Đang calibrate giọng nói...")

        # 5 câu calibration đa dạng
        cal_texts = [
            CALIBRATION_TEXT,
            "hôm nay thời tiết rất đẹp, tôi muốn đi dạo ở ngoài công viên thành phố.",
            "chào mừng quý vị và các bạn đến với chương trình phát thanh tiếng Việt.",
            "trong cuộc sống hằng ngày, chúng ta luôn cần sự kiên nhẫn và lòng quyết tâm.",
            "khoa học và công nghệ đang thay đổi thế giới một cách nhanh chóng và mạnh mẽ.",
        ]

        best_audio = None
        best_score = -1
        best_transcript = ""
        best_coverage = 0
        best_sr = 24000

        profile_id = str(uuid.uuid4())[:8]

        # 2-phase strategy: Phase 1 nfe=64 (high quality), Phase 2 nfe=32 (fallback)
        phases = [
            {"nfe": 64, "speed": 0.95, "attempts": 3, "label": "cao"},
            {"nfe": 32, "speed": 1.0,  "attempts": 2, "label": "chuẩn"},
        ]

        total_attempts = sum(p["attempts"] for p in phases)
        attempt_idx = 0

        for phase in phases:
            nfe = phase["nfe"]
            spd = phase["speed"]

            for attempt in range(phase["attempts"]):
                attempt_idx += 1
                cal_text = cal_texts[(attempt_idx - 1) % len(cal_texts)]
                progress_pct = 25 + int(50 * attempt_idx / total_attempts)
                _synthesis_progress[task_id].update(
                    progress=min(progress_pct, 75),
                    message=f"🔊 Calibration {attempt_idx}/{total_attempts} (nfe={nfe})..."
                )

                output_tmp = OUTPUT_DIR / f"_f5_cal_{profile_id}_{attempt_idx}.wav"
                try:
                    # GPU memory cleanup trước mỗi attempt
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                    seed = _random.randint(0, 2**31)
                    wav, sr, _ = f5engine.infer(
                        ref_file=str(cleaned_path),
                        ref_text=ref_text_final,
                        gen_text=cal_text,
                        nfe_step=nfe,
                        speed=spd,
                        remove_silence=True,
                        cross_fade_duration=0.15,
                        seed=seed,
                    )

                    if wav is None or len(wav) == 0:
                        # Fallback: đọc từ file nếu infer trả None
                        if output_tmp.exists():
                            data, samplerate = sf.read(str(output_tmp))
                            wav = data
                            sr = samplerate
                        else:
                            logger.warning(f"   ⚠️ Attempt {attempt_idx}: empty output")
                            continue

                    duration = len(wav) / sr
                    if duration < 0.5:
                        logger.warning(f"   ⚠️ Attempt {attempt_idx}: quá ngắn ({duration:.1f}s)")
                        continue

                    # Whisper verification — wrapped in try/catch cho ổn định
                    overlap = 0.0
                    transcript = ""
                    try:
                        import librosa
                        audio_16k = librosa.resample(
                            wav.astype(np.float32), orig_sr=sr, target_sr=16000
                        )
                        tmp_path = PROFILES_DIR / f"_tmp_f5cal_{attempt_idx}.wav"
                        sf.write(str(tmp_path), audio_16k, 16000)
                        transcript = auto_transcribe(tmp_path) or ""
                        tmp_path.unlink(missing_ok=True)

                        if transcript:
                            cal_words = set(
                                cal_text.lower().replace(".", "").replace(",", "").split()
                            )
                            trans_words = set(
                                transcript.lower().replace(".", "").replace(",", "").split()
                            )
                            overlap = len(cal_words & trans_words) / max(len(cal_words), 1)
                    except Exception as we:
                        logger.warning(f"   ⚠️ Whisper skip (attempt {attempt_idx}): {we}")
                        overlap = 0.5  # Assume decent nếu Whisper lỗi

                    # Scoring
                    expected_dur = len(cal_text) * (0.085 if spd < 1.0 else 0.08)
                    dur_score = 1.0 - min(abs(duration - expected_dur) / max(expected_dur, 1), 1.0)
                    rms = np.sqrt(np.mean(wav ** 2))
                    snr_score = min(rms / 0.02, 1.0) if rms > 0 else 0
                    score = overlap * 0.6 + dur_score * 0.2 + snr_score * 0.2

                    logger.info(
                        f"   🔄 Attempt {attempt_idx} (nfe={nfe}): {duration:.1f}s, "
                        f"coverage={overlap:.0%}, dur={dur_score:.2f}, "
                        f"snr={snr_score:.2f}, total={score:.2f}"
                    )

                    if score > best_score:
                        best_score = score
                        best_audio = wav.copy()  # Copy để tránh reference bị overwrite
                        best_sr = sr
                        best_transcript = transcript
                        best_coverage = overlap

                    # Đạt chất lượng tốt → dừng sớm
                    if overlap >= 0.80 and dur_score >= 0.4:
                        logger.info(f"   ✅ Đạt chất lượng tốt (coverage={overlap:.0%})")
                        break

                except Exception as e:
                    logger.warning(f"   ⚠️ Attempt {attempt_idx} failed: {e}")
                    continue
                finally:
                    # Cleanup temp files
                    output_tmp.unlink(missing_ok=True)

            # Nếu đã có kết quả tốt, skip phase tiếp theo
            if best_audio is not None and best_coverage >= 0.80:
                break

        # Fallback: nếu tất cả attempts thất bại, thử 1 lần cuối đơn giản nhất
        if best_audio is None:
            logger.warning("   🔄 Tất cả attempts thất bại, thử fallback đơn giản...")
            _synthesis_progress[task_id].update(progress=78, message="🔄 Đang thử phương pháp dự phòng...")
            try:
                wav, sr, _ = f5engine.infer(
                    ref_file=str(cleaned_path),
                    ref_text=ref_text_final,
                    gen_text=CALIBRATION_TEXT,
                    nfe_step=16,  # Nhanh nhất, ít lỗi nhất
                    speed=1.0,
                    seed=42,
                )
                if wav is not None and len(wav) > 0:
                    best_audio = wav.copy()
                    best_sr = sr
                    best_score = 0.3
                    best_coverage = 0.0
                    logger.info(f"   ✅ Fallback thành công: {len(wav)/sr:.1f}s")
            except Exception as fe:
                logger.error(f"   ❌ Fallback cũng thất bại: {fe}")

        if best_audio is None:
            _synthesis_progress[task_id].update(
                status="error", progress=100,
                message="❌ Không thể tạo giọng mẫu — thử audio mẫu khác"
            )
            raise RuntimeError("Không thể tạo calibration audio với F5-TTS")

        # Bước 4: Lưu calibration audio
        _synthesis_progress[task_id].update(progress=85, message="💾 Đang lưu giọng mẫu...")

        # Normalize + soft-limiter trước khi lưu
        max_val = np.max(np.abs(best_audio))
        if max_val > 0:
            best_audio = (best_audio * (0.95 / max_val)).astype(np.float32)

        cal_filename = f"calibration_{profile_id}.wav"
        cal_path = PROFILES_DIR / cal_filename
        sf.write(str(cal_path), best_audio, best_sr)

        # Bước 5: Lưu profile JSON
        profile_data = {
            "id": profile_id,
            "name": profile_name,
            "ref_codes": [],
            "ref_text": ref_text_final,
            "calibration_audio": cal_filename,
            "source_audio": trimmed_filename,
            "created_at": datetime.now().isoformat(),
            "quality_score": round(best_score, 3),
            "coverage": round(best_coverage, 3),
            "engine": "f5-tts",
        }

        profile_path = PROFILES_DIR / f"{profile_id}.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json_mod.dump(profile_data, f, ensure_ascii=False, indent=2)

        _synthesis_progress[task_id].update(
            status="done", progress=100,
            message=f"✅ Giọng mẫu '{profile_name}' đã sẵn sàng!"
        )

        logger.info(f"✅ Đã tạo F5-TTS profile: {profile_name} ({profile_id}) — score: {best_score:.2f}")

        return {
            "id": profile_id,
            "name": profile_name,
            "calibration_audio": cal_filename,
            "codes_count": 0,
            "created_at": profile_data["created_at"],
            "engine": "f5-tts",
            "quality_score": round(best_score, 3),
            "coverage": round(best_coverage, 3),
            "task_id": task_id,
        }

    except Exception as e:
        if task_id in _synthesis_progress:
            _synthesis_progress[task_id].update(
                status="error", progress=100,
                message=f"❌ {str(e)[:60]}"
            )
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
                    "engine": data.get("engine", ""),
                    "quality_score": data.get("quality_score"),
                    "coverage": data.get("coverage"),
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


# ── Vietnamese Number-to-Words ───────────────────────────────────
_VN_ONES = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_VN_UNITS_MAP = {
    "km/h": "ki-lô-mét trên giờ", "km": "ki-lô-mét", "m/s": "mét trên giây",
    "m²": "mét vuông", "m³": "mét khối", "m": "mét", "cm": "xen-ti-mét",
    "mm": "mi-li-mét", "kg": "ki-lô-gam", "g": "gam", "mg": "mi-li-gam",
    "l": "lít", "ml": "mi-li-lít", "ha": "héc-ta",
    "°C": "độ C", "°F": "độ F",
}
_VN_ABBREV = {
    "TP.HCM": "Thành phố Hồ Chí Minh", "TP.": "Thành phố",
    "PGĐ": "Phó Giám đốc", "GĐ": "Giám đốc",
    "TS.": "Tiến sĩ", "ThS.": "Thạc sĩ", "PGS.": "Phó Giáo sư",
    "GS.": "Giáo sư", "CN.": "Cử nhân",
    "UBND": "Ủy ban Nhân dân", "HĐND": "Hội đồng Nhân dân",
    "VNĐ": "Việt Nam đồng", "USD": "đô la Mỹ",
}


def _num_to_words_vi(n: int) -> str:
    """Chuyển số nguyên thành chữ tiếng Việt (hỗ trợ đến hàng tỷ)."""
    if n < 0:
        return "âm " + _num_to_words_vi(-n)
    if n == 0:
        return "không"

    parts = []
    if n >= 1_000_000_000:
        parts.append(_num_to_words_vi(n // 1_000_000_000) + " tỷ")
        n %= 1_000_000_000
    if n >= 1_000_000:
        parts.append(_num_to_words_vi(n // 1_000_000) + " triệu")
        n %= 1_000_000
    if n >= 1_000:
        parts.append(_num_to_words_vi(n // 1_000) + " nghìn")
        n %= 1_000
    if n >= 100:
        parts.append(_VN_ONES[n // 100] + " trăm")
        n %= 100
        if n > 0 and n < 10:
            parts.append("lẻ")
    if n >= 10:
        tens = n // 10
        if tens == 1:
            parts.append("mười")
        else:
            parts.append(_VN_ONES[tens] + " mươi")
        n %= 10
        if n == 1 and tens > 1:
            parts.append("mốt")
            n = 0
        elif n == 5 and tens >= 1:
            parts.append("lăm")
            n = 0
        elif n == 4 and tens >= 2:
            parts.append("tư")
            n = 0
    if n > 0:
        parts.append(_VN_ONES[n])

    return " ".join(parts)


def normalize_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt cho TTS:
    - Số → chữ: 123 → một trăm hai mươi ba
    - Tiền tệ: 50.000đ → năm mươi nghìn đồng
    - Phần trăm: 80% → tám mươi phần trăm
    - Đơn vị: km/h, kg, cm...
    - Viết tắt: TP.HCM, GS., TS....
    - Ngày tháng: 25/03/2024
    """
    # 1. Viết tắt (trước khi xử lý khác)
    for abbr, full in _VN_ABBREV.items():
        text = text.replace(abbr, full)

    # 2. Ngày tháng: 25/03/2024 hoặc 25/3
    def _replace_date(m):
        day, month = int(m.group(1)), int(m.group(2))
        year = m.group(3)
        result = f"ngày {_num_to_words_vi(day)} tháng {_num_to_words_vi(month)}"
        if year:
            result += f" năm {_num_to_words_vi(int(year))}"
        return result
    text = re.sub(r'(\d{1,2})/(\d{1,2})(?:/(\d{4}))?', _replace_date, text)

    # 3. Tiền tệ: 50.000đ, 1.234.567đ
    def _replace_currency(m):
        num_str = m.group(1).replace(".", "")
        unit = m.group(2)
        n = int(num_str)
        words = _num_to_words_vi(n)
        if unit.lower() in ("đ", "đồng", "vnd"):
            return words + " đồng"
        return words + " " + unit
    text = re.sub(r'([\d.]+)(đ|đồng|VND)\b', _replace_currency, text)

    # 4. Phần trăm: 80%
    def _replace_percent(m):
        return _num_to_words_vi(int(m.group(1))) + " phần trăm"
    text = re.sub(r'(\d+)%', _replace_percent, text)

    # 5. Số + đơn vị: 120km/h, 5kg, 30°C
    def _replace_unit(m):
        num = int(m.group(1))
        unit = m.group(2)
        unit_word = _VN_UNITS_MAP.get(unit, unit)
        return _num_to_words_vi(num) + " " + unit_word
    units_pattern = "|".join(re.escape(u) for u in sorted(_VN_UNITS_MAP.keys(), key=len, reverse=True))
    text = re.sub(rf'(\d+)\s*({units_pattern})\b', _replace_unit, text)

    # 6. Số thập phân: 3.14, 2,5
    def _replace_decimal(m):
        integer_part = int(m.group(1))
        decimal_part = m.group(2)
        result = _num_to_words_vi(integer_part) + " phẩy"
        for digit in decimal_part:
            result += " " + _VN_ONES[int(digit)]
        return result
    text = re.sub(r'(\d+)[.,](\d{1,3})(?!\d)', _replace_decimal, text)

    # 7. Số nguyên còn lại (có dấu chấm phân cách hàng nghìn: 1.234.567)
    def _replace_grouped_number(m):
        num_str = m.group(0).replace(".", "")
        return _num_to_words_vi(int(num_str))
    text = re.sub(r'\d{1,3}(?:\.\d{3})+', _replace_grouped_number, text)

    # 8. Số nguyên đơn giản
    def _replace_plain_number(m):
        return _num_to_words_vi(int(m.group(0)))
    text = re.sub(r'\b\d+\b', _replace_plain_number, text)

    return text


def preprocess_vietnamese_text(text: str) -> str:
    """
    Tiền xử lý văn bản tiếng Việt cho TTS:
    - Chuẩn hóa khoảng trắng
    - Chuyển số/ký hiệu thành chữ
    - Thêm dấu chấm cuối câu nếu thiếu
    - Tách dòng thành câu riêng biệt
    """
    # Chuẩn hóa xuống hàng: thay \n bằng dấu chấm + space
    text = re.sub(r'\n+', '. ', text)

    # Chuẩn hóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # Chuyển số, đơn vị, viết tắt thành chữ
    text = normalize_vietnamese_text(text)

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
    """Lấy hoặc khởi tạo F5-TTS engine (singleton) — Vietnamese model"""
    global _f5tts_engine
    if _f5tts_engine is None:
        logger.info("🚀 Đang khởi tạo F5-TTS engine (Vietnamese)...")
        try:
            from f5_tts.api import F5TTS
            from cached_path import cached_path

            # Tải checkpoint và vocab tiếng Việt từ HuggingFace
            vi_ckpt = str(cached_path("hf://toandev/F5-TTS-Vietnamese/model_last.pt"))
            vi_vocab = str(cached_path("hf://toandev/F5-TTS-Vietnamese/vocab.txt"))
            logger.info(f"   📦 Vietnamese ckpt: {vi_ckpt}")
            logger.info(f"   📦 Vietnamese vocab: {vi_vocab}")

            _f5tts_engine = F5TTS(
                model="F5TTS_Base",
                ckpt_file=vi_ckpt,
                vocab_file=vi_vocab,
            )
            logger.info("✅ F5-TTS Vietnamese engine đã sẵn sàng!")
        except Exception as e:
            logger.error(f"❌ Lỗi khởi tạo F5-TTS: {e}")
            raise RuntimeError(f"Lỗi khởi tạo F5-TTS: {e}")
    return _f5tts_engine


# ── Synthesis Progress Tracking ──────────────────────────────────
_synthesis_progress = {}  # task_id -> {status, progress, message}


@app.get("/api/synthesis-progress/{task_id}")
async def get_synthesis_progress(task_id: str):
    """Lấy tiến trình tổng hợp giọng nói"""
    if task_id in _synthesis_progress:
        return _synthesis_progress[task_id]
    return {"status": "unknown", "progress": 0, "message": "Không tìm thấy task"}


# Quality presets: nfe_step mapping
_QUALITY_PRESETS = {
    "fast": 16,      # Nhanh, chất lượng cơ bản
    "standard": 32,  # Cân bằng (mặc định)
    "high": 64,      # Chất lượng cao, chậm hơn
}


# ── Synthesize with F5-TTS (High Quality Voice Cloning) ─────────
@app.post("/api/synthesize-f5")
async def synthesize_f5tts(
    text: str = Form(...),
    trimmed_filename: str = Form(default=""),
    voice_profile_id: str = Form(default=""),
    ref_text: str = Form(default=""),
    speed: float = Form(default=1.0),
    quality: str = Form(default="standard"),
):
    """
    Tổng hợp giọng nói với F5-TTS — engine chất lượng cao.
    - quality: fast / standard / high
    - speed: 0.5 — 2.0
    - Tự động: chunking + crossfade + Whisper verification + silence removal
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản")

    # Clamp speed
    speed = max(0.5, min(2.0, speed))
    nfe_step = _QUALITY_PRESETS.get(quality, 32)

    # Task ID cho progress tracking
    task_id = str(uuid.uuid4())[:8]
    _synthesis_progress[task_id] = {
        "status": "processing", "progress": 0,
        "message": "Đang chuẩn bị...", "task_id": task_id,
    }

    try:
        f5engine = get_f5tts_engine()

        # Bước 1: Tiền xử lý text (bao gồm number normalization)
        _synthesis_progress[task_id].update(progress=10, message="Đang xử lý văn bản...")
        processed_text = preprocess_vietnamese_text(text)
        start_time = time.time()

        logger.info(f"   📝 Text gốc: \"{text[:80]}...\"")
        logger.info(f"   📝 Sau chuẩn hóa ({len(processed_text)} chars): \"{processed_text[:100]}...\"")

        # Bước 2: Resolve reference audio
        _synthesis_progress[task_id].update(progress=15, message="Đang tải giọng mẫu...")
        ref_audio_path = None

        if voice_profile_id.strip():
            import json as json_mod
            profile_path = PROFILES_DIR / f"{voice_profile_id}.json"
            if not profile_path.exists():
                raise HTTPException(status_code=404, detail="Voice profile không tìm thấy")
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json_mod.load(f)
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

        logger.info(f"   🎙️ Ref text: \"{ref_text[:60]}\"")
        logger.info(f"   🚀 Speed: {speed}, Quality: {quality} (nfe={nfe_step})")

        # Bước 3: Tổng hợp — F5-TTS tự chunking + crossfade
        _synthesis_progress[task_id].update(
            progress=20,
            message=f"Đang tổng hợp giọng nói ({quality})..."
        )

        import random as _random

        best_wav = None
        best_duration = 0
        best_coverage = 0
        expected_duration = len(processed_text) * 0.08 / speed
        min_acceptable = expected_duration * 0.4

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            seed = _random.randint(0, 2**31)
            progress_pct = 20 + int(50 * (attempt + 1) / MAX_RETRIES)
            _synthesis_progress[task_id].update(
                progress=min(progress_pct, 70),
                message=f"Đang tổng hợp (lần {attempt+1}/{MAX_RETRIES})..."
            )

            try:
                wav, sr, _ = f5engine.infer(
                    ref_file=ref_audio_path,
                    ref_text=ref_text,
                    gen_text=processed_text,
                    speed=speed,
                    cross_fade_duration=0.15,
                    nfe_step=nfe_step,
                    remove_silence=True,  # Tự động xóa khoảng lặng
                    seed=seed,
                )

                if wav is not None and len(wav) > 0:
                    duration = len(wav) / sr
                    logger.info(
                        f"   🔊 Attempt {attempt+1}: {duration:.1f}s "
                        f"(expected≥{min_acceptable:.1f}s) seed={seed}"
                    )

                    # Whisper post-verification
                    coverage = 0.0
                    if duration >= 1.0:
                        _synthesis_progress[task_id].update(
                            progress=min(progress_pct + 5, 75),
                            message="Đang kiểm tra chất lượng (Whisper)..."
                        )
                        try:
                            import librosa
                            audio_16k = librosa.resample(
                                wav.astype(np.float32), orig_sr=sr, target_sr=16000
                            )
                            tmp_verify = OUTPUT_DIR / f"_verify_{task_id}_{attempt}.wav"
                            sf.write(str(tmp_verify), audio_16k, 16000)
                            transcript = auto_transcribe(tmp_verify)
                            tmp_verify.unlink(missing_ok=True)

                            if transcript:
                                # So sánh word coverage
                                input_words = set(
                                    processed_text.lower()
                                    .replace(".", "").replace(",", "")
                                    .replace("!", "").replace("?", "")
                                    .split()
                                )
                                trans_words = set(
                                    transcript.lower()
                                    .replace(".", "").replace(",", "")
                                    .replace("!", "").replace("?", "")
                                    .split()
                                )
                                coverage = len(input_words & trans_words) / max(len(input_words), 1)
                                logger.info(
                                    f"   📋 Whisper coverage: {coverage:.0%} "
                                    f"| \"{transcript[:60]}...\""
                                )
                        except Exception as e:
                            logger.warning(f"   ⚠️ Whisper verify failed: {e}")
                            coverage = 1.0  # Skip verification nếu lỗi

                    # Giữ kết quả tốt nhất
                    if duration > best_duration or coverage > best_coverage:
                        best_duration = duration
                        best_wav = wav
                        best_coverage = coverage

                    # Đủ dài VÀ coverage tốt → chấp nhận
                    if duration >= min_acceptable and coverage >= 0.6:
                        logger.info(f"   ✅ Đạt chất lượng tốt (coverage={coverage:.0%})")
                        break
                    elif duration >= min_acceptable and coverage < 0.6 and attempt < MAX_RETRIES - 1:
                        logger.info(f"   🔄 Coverage thấp ({coverage:.0%}), thử lại...")
                else:
                    logger.warning(f"   ⚠️ Attempt {attempt+1}: empty output")

            except Exception as e:
                logger.warning(f"   ⚠️ Attempt {attempt+1} failed: {e}")
                continue

        if best_wav is None:
            _synthesis_progress[task_id].update(
                status="error", progress=100, message="Không thể tổng hợp audio"
            )
            raise RuntimeError("Không thể tổng hợp audio")

        # Bước 4: Post-processing
        _synthesis_progress[task_id].update(progress=85, message="Đang hoàn thiện audio...")

        audio_output = best_wav
        sample_rate = 24000

        # Normalize volume
        if isinstance(audio_output, np.ndarray) and len(audio_output) > 0:
            max_val = np.max(np.abs(audio_output))
            if max_val > 0:
                audio_output = (audio_output * (0.95 / max_val)).astype(np.float32)

        # Lưu output
        output_id = str(uuid.uuid4())[:8]
        output_filename = f"output_{output_id}.wav"
        output_path = OUTPUT_DIR / output_filename
        sf.write(str(output_path), audio_output, sample_rate)

        elapsed = time.time() - start_time
        total_duration = len(audio_output) / sample_rate

        _synthesis_progress[task_id].update(
            status="done", progress=100,
            message=f"Hoàn tất! {total_duration:.1f}s audio trong {elapsed:.1f}s"
        )

        logger.info(
            f"✅ F5-TTS thành công: {total_duration:.1f}s audio "
            f"trong {elapsed:.1f}s, coverage={best_coverage:.0%} → {output_filename}"
        )

        return {
            "output_filename": output_filename,
            "processing_time": round(elapsed, 2),
            "text_length": len(processed_text),
            "audio_duration": round(total_duration, 2),
            "engine": "f5-tts",
            "quality": quality,
            "speed": speed,
            "coverage": round(best_coverage, 2),
            "task_id": task_id,
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
        reload_excludes=["data", "outputs", "frontend", "*.wav", "*.json", "*.log", "voices.json"],
        log_level="info",
    )
