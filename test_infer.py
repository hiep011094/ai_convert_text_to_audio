import sys
from pathlib import Path

# Thêm đường dẫn vào sys.path để import
sys.path.append(str(Path(__file__).parent))

from vieneu import Vieneu
engine = Vieneu(backbone_device="cuda", codec_device="cuda")

# Test infer
try:
    print("Testing F5-TTS inference...")
    wav, sr, _ = engine.infer(
        ref_file="voice_profiles/calibration_08368870.wav",
        ref_text="xin chào",
        gen_text="thử nghiệm hệ thống",
        speed=1.0,
        cross_fade_duration=0.15,
        nfe_step=16,
        remove_silence=True
    )
    if wav is not None:
        print(f"Success! {len(wav)/sr}s generated")
    else:
        print("wav is None!")
except Exception as e:
    import traceback
    traceback.print_exc()
