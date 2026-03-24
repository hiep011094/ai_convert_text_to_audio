"""Quick test script for all API endpoints"""
import requests
import json
import time

API = "http://127.0.0.1:8000"
results = []

def test(name, func):
    try:
        ok, detail = func()
        status = "✅ PASS" if ok else "❌ FAIL"
        results.append((name, status, detail))
        print(f"  {status}: {name} - {detail}")
    except Exception as e:
        results.append((name, "❌ ERROR", str(e)))
        print(f"  ❌ ERROR: {name} - {e}")

print("=" * 60)
print("VN-VoiceClone Pro - API Tests")
print("=" * 60)

# 1. Health Check
def test_health():
    r = requests.get(f"{API}/api/health", timeout=5)
    d = r.json()
    return d["status"] == "ok", f"status={d['status']}, engine_ready={d['engine_ready']}"
test("[1] GET /api/health", test_health)

# 2. Upload Audio
upload_data = {}
def test_upload():
    with open("test_audio.wav", "rb") as f:
        r = requests.post(f"{API}/api/upload", files={"file": ("test.wav", f, "audio/wav")}, timeout=10)
    d = r.json()
    upload_data.update(d)
    return r.status_code == 200 and d.get("duration", 0) > 0, f"file_id={d.get('file_id')}, duration={d.get('duration')}s, size={d.get('size')}B"
test("[2] POST /api/upload", test_upload)

# 3. Serve Uploaded Audio
def test_serve_upload():
    fn = upload_data.get("filename", "")
    r = requests.get(f"{API}/api/audio/uploads/{fn}", timeout=5)
    return r.status_code == 200 and len(r.content) > 0, f"filename={fn}, content_length={len(r.content)}"
test("[3] GET /api/audio/uploads/{filename}", test_serve_upload)

# 4. Upload invalid format
def test_upload_invalid():
    r = requests.post(f"{API}/api/upload", files={"file": ("test.txt", b"hello", "text/plain")}, timeout=5)
    return r.status_code == 400, f"status_code={r.status_code}"
test("[4] POST /api/upload (invalid format)", test_upload_invalid)

# 5. Trim Audio
trim_data = {}
def test_trim():
    fn = upload_data.get("filename", "")
    r = requests.post(f"{API}/api/trim", data={"filename": fn, "start": 0, "end": 5}, timeout=10)
    d = r.json()
    trim_data.update(d)
    return r.status_code == 200 and d.get("trimmed_filename"), f"trimmed={d.get('trimmed_filename')}, duration={d.get('duration')}s"
test("[5] POST /api/trim (0-5s)", test_trim)

# 6. Trim - too short (< 1s)
def test_trim_short():
    fn = upload_data.get("filename", "")
    r = requests.post(f"{API}/api/trim", data={"filename": fn, "start": 0, "end": 0.5}, timeout=5)
    return r.status_code == 400, f"status_code={r.status_code}"
test("[6] POST /api/trim (too short < 1s)", test_trim_short)

# 7. Trim - too long (> 30s)
def test_trim_long():
    fn = upload_data.get("filename", "")
    r = requests.post(f"{API}/api/trim", data={"filename": fn, "start": 0, "end": 31}, timeout=5)
    return r.status_code == 400, f"status_code={r.status_code}"
test("[7] POST /api/trim (too long > 30s)", test_trim_long)

# 8. Serve Trimmed Audio
def test_serve_trimmed():
    fn = trim_data.get("trimmed_filename", "")
    r = requests.get(f"{API}/api/audio/trimmed/{fn}", timeout=5)
    return r.status_code == 200 and len(r.content) > 0, f"trimmed_filename={fn}, size={len(r.content)}"
test("[8] GET /api/audio/trimmed/{filename}", test_serve_trimmed)

# 9. List Voices
def test_voices():
    r = requests.get(f"{API}/api/voices", timeout=10)
    d = r.json()
    voices = d.get("voices", [])
    return r.status_code == 200, f"found {len(voices)} voices"
test("[9] GET /api/voices", test_voices)

# 10. Synthesize
def test_synthesize():
    fn = trim_data.get("trimmed_filename", "")
    st = time.time()
    r = requests.post(f"{API}/api/synthesize", data={
        "trimmed_filename": fn,
        "text": "Xin chào, đây là bài test tổng hợp giọng nói.",
        "ref_text": "",
    }, timeout=300)
    elapsed = time.time() - st
    if r.status_code == 200:
        d = r.json()
        return True, f"output={d.get('output_filename')}, time={d.get('processing_time')}s (total={elapsed:.1f}s)"
    else:
        return False, f"status_code={r.status_code}, detail={r.text[:200]}"
test("[10] POST /api/synthesize", test_synthesize)

# 11. Serve + Download Output
def test_serve_output():
    # Find output filename from synthesize
    fn = trim_data.get("trimmed_filename", "")
    # Re-get from synthesize result if available
    r = requests.get(f"{API}/api/audio/outputs/nonexistent.wav", timeout=5)
    return r.status_code == 404, f"404 for nonexistent file: status={r.status_code}"
test("[11] GET /api/audio/outputs (404 test)", test_serve_output)

# 12. Download 404
def test_download_404():
    r = requests.get(f"{API}/api/download/nonexistent.wav", timeout=5)
    return r.status_code == 404, f"status={r.status_code}"
test("[12] GET /api/download (404 test)", test_download_404)

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
passed = sum(1 for _, s, _ in results if "PASS" in s)
failed = sum(1 for _, s, _ in results if "FAIL" in s or "ERROR" in s)
print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
for name, status, detail in results:
    print(f"  {status}: {name}")
print("=" * 60)
