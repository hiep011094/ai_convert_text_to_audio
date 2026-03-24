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

# ── Cấu hình đường dẫn ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TRIMMED_DIR = BASE_DIR / "trimmed"

for d in [UPLOAD_DIR, OUTPUT_DIR, TRIMMED_DIR]:
    d.mkdir(exist_ok=True)

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
            from vieneu import Vieneu
            tts_engine = Vieneu()
            logger.info("✅ VieNeu-TTS engine đã sẵn sàng!")
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
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(source_path))

        # Cắt (pydub dùng milliseconds)
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        trimmed = audio[start_ms:end_ms]

        # Lưu dạng WAV (VieNeu-TTS yêu cầu WAV)
        trim_id = str(uuid.uuid4())[:8]
        trim_filename = f"trimmed_{trim_id}.wav"
        trim_path = TRIMMED_DIR / trim_filename
        trimmed.export(str(trim_path), format="wav")

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


# ── Synthesize Voice ─────────────────────────────────────────────
@app.post("/api/synthesize")
async def synthesize_voice(
    trimmed_filename: str = Form(...),
    text: str = Form(...),
    ref_text: str = Form(default=""),
):
    """
    Tổng hợp giọng nói từ văn bản + mẫu audio reference.
    Sử dụng VieNeu-TTS zero-shot voice cloning.
    """
    # Validate
    if not text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản")

    ref_path = TRIMMED_DIR / trimmed_filename
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail="File mẫu giọng không tìm thấy")

    try:
        engine = get_tts_engine()

        logger.info(f"🎤 Đang tổng hợp giọng nói...")
        logger.info(f"   📝 Văn bản: {text[:100]}...")
        logger.info(f"   🎵 Mẫu: {trimmed_filename}")

        start_time = time.time()

        # Gọi VieNeu-TTS với voice cloning
        audio_output = engine.infer(
            text=text,
            ref_audio=str(ref_path),
            ref_text=ref_text if ref_text.strip() else None,
        )

        # Lưu kết quả
        output_id = str(uuid.uuid4())[:8]
        output_filename = f"output_{output_id}.wav"
        output_path = OUTPUT_DIR / output_filename

        engine.save(audio_output, str(output_path))

        elapsed = time.time() - start_time
        logger.info(f"✅ Tổng hợp thành công trong {elapsed:.1f}s → {output_filename}")

        return {
            "output_filename": output_filename,
            "processing_time": round(elapsed, 2),
            "text_length": len(text),
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
