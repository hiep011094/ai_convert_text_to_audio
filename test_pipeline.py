"""Test the full VN-VoiceClone Pro pipeline"""
import requests
import time

API = "http://127.0.0.1:12345"

print("=== VN-VoiceClone Pro Pipeline Test ===")

# 1. Health Check
print("[1/5] Health Check...")
r = requests.get(API + "/api/health")
h = r.json()
print("  Status:", h["status"], "Engine:", h["engine_ready"])
assert h["status"] == "ok" and h["engine_ready"] == True
print("  PASS")

# 2. Upload
print("[2/5] Upload Audio...")
with open("test_audio.wav", "rb") as f:
    r = requests.post(API + "/api/upload", files={"file": ("test.wav", f, "audio/wav")})
u = r.json()
print("  File:", u["file_id"], "Duration:", u["duration"], "s")
assert u["duration"] > 0
print("  PASS")

# 3. Trim
print("[3/5] Trim Audio 0-5s...")
r = requests.post(API + "/api/trim", data={"filename": u["filename"], "start": 0, "end": 5})
t = r.json()
print("  Trimmed:", t["trimmed_filename"])
print("  PASS")

# 4. Synthesize
print("[4/5] Synthesize (CPU, may take 30-60s)...")
st = time.time()
r = requests.post(API + "/api/synthesize", data={
    "trimmed_filename": t["trimmed_filename"],
    "text": "Xin chao, toi la mot tro ly ao duoc tao boi VN-VoiceClone Pro.",
    "ref_text": "",
}, timeout=300)
elapsed = time.time() - st
if r.status_code == 200:
    s = r.json()
    print("  Output:", s["output_filename"])
    print("  Processing:", s["processing_time"], "s (total:", round(elapsed, 1), "s)")
    print("  PASS")

    # 5. Download
    print("[5/5] Download output...")
    r2 = requests.get(API + "/api/download/" + s["output_filename"])
    with open("test_output.wav", "wb") as f:
        f.write(r2.content)
    print("  Size:", len(r2.content), "bytes -> test_output.wav")
    print("  PASS")
    print("=== ALL TESTS PASSED ===")
else:
    print("  FAIL:", r.status_code, r.text[:300])
