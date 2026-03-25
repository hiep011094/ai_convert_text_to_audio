const API_BASE = "http://localhost:12345";

export interface UploadResult {
  file_id: string;
  filename: string;
  original_name: string;
  duration: number;
  sample_rate: number;
  channels: number;
  size: number;
}

export interface TrimResult {
  trimmed_filename: string;
  duration: number;
  start: number;
  end: number;
}

export interface SynthesisResult {
  output_filename: string;
  processing_time: number;
  text_length: number;
  audio_duration?: number;
  quality?: string;
  speed?: number;
  coverage?: number;
  task_id?: string;
  engine?: string;
}

export interface SynthesisProgress {
  status: string;
  progress: number;
  message: string;
  task_id?: string;
}

export interface VoicePreset {
  description: string;
  id: string;
}

export interface VoiceProfile {
  id: string;
  name: string;
  calibration_audio: string;
  codes_count: number;
  created_at: string;
  engine?: string;
  quality_score?: number;
  coverage?: number;
  task_id?: string;
}

export async function uploadAudio(file: File): Promise<UploadResult> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Lỗi upload" }));
    throw new Error(err.detail || "Lỗi khi upload file");
  }
  return res.json();
}

export async function trimAudio(
  filename: string,
  start: number,
  end: number
): Promise<TrimResult> {
  const formData = new FormData();
  formData.append("filename", filename);
  formData.append("start", start.toString());
  formData.append("end", end.toString());

  const res = await fetch(`${API_BASE}/api/trim`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Lỗi cắt audio" }));
    throw new Error(err.detail || "Lỗi khi cắt audio");
  }
  return res.json();
}

export async function synthesizeVoice(
  text: string,
  options?: {
    trimmedFilename?: string;
    refText?: string;
    voiceProfileId?: string;
    engine?: "vieneu" | "f5-tts";
    speed?: number;
    quality?: "fast" | "standard" | "high";
  }
): Promise<SynthesisResult> {
  const formData = new FormData();
  formData.append("text", text);
  if (options?.trimmedFilename) {
    formData.append("trimmed_filename", options.trimmedFilename);
  }
  if (options?.refText) {
    formData.append("ref_text", options.refText);
  }
  if (options?.voiceProfileId) {
    formData.append("voice_profile_id", options.voiceProfileId);
  }
  if (options?.speed !== undefined) {
    formData.append("speed", options.speed.toString());
  }
  if (options?.quality) {
    formData.append("quality", options.quality);
  }

  // Choose API endpoint based on engine
  const endpoint = options?.engine === "f5-tts"
    ? `${API_BASE}/api/synthesize-f5`
    : `${API_BASE}/api/synthesize`;

  const res = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Lỗi tổng hợp" }));
    throw new Error(err.detail || "Lỗi khi tổng hợp giọng nói");
  }
  return res.json();
}

export async function getSynthesisProgress(taskId: string): Promise<SynthesisProgress> {
  const res = await fetch(`${API_BASE}/api/synthesis-progress/${taskId}`);
  return res.json();
}

// ── Voice Profile APIs ──────────────────────────────────────────

export async function createVoiceProfile(
  trimmedFilename: string,
  profileName: string,
  refText?: string,
  engine?: "vieneu" | "f5-tts"
): Promise<VoiceProfile> {
  const formData = new FormData();
  formData.append("trimmed_filename", trimmedFilename);
  formData.append("profile_name", profileName);
  if (refText) {
    formData.append("ref_text", refText);
  }

  const endpoint = engine === "f5-tts"
    ? `${API_BASE}/api/create-voice-profile-f5`
    : `${API_BASE}/api/create-voice-profile`;

  const res = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res
      .json()
      .catch(() => ({ detail: "Lỗi tạo voice profile" }));
    throw new Error(err.detail || "Lỗi khi tạo voice profile");
  }
  return res.json();
}

export async function listVoiceProfiles(): Promise<VoiceProfile[]> {
  const res = await fetch(`${API_BASE}/api/voice-profiles`);
  const data = await res.json();
  return data.profiles || [];
}

export async function deleteVoiceProfile(profileId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/voice-profiles/${profileId}`, {
    method: "DELETE",
  });

  if (!res.ok) {
    const err = await res
      .json()
      .catch(() => ({ detail: "Lỗi xóa profile" }));
    throw new Error(err.detail || "Lỗi khi xóa voice profile");
  }
}

export async function checkHealth(): Promise<{
  status: string;
  engine_ready: boolean;
}> {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

export async function listVoices(): Promise<VoicePreset[]> {
  const res = await fetch(`${API_BASE}/api/voices`);
  const data = await res.json();
  return data.voices || [];
}

export function getUploadedAudioUrl(filename: string): string {
  return `${API_BASE}/api/audio/uploads/${filename}`;
}

export function getTrimmedAudioUrl(filename: string): string {
  return `${API_BASE}/api/audio/trimmed/${filename}`;
}

export function getOutputAudioUrl(filename: string): string {
  return `${API_BASE}/api/audio/outputs/${filename}`;
}

export function getProfileAudioUrl(filename: string): string {
  return `${API_BASE}/api/audio/profiles/${filename}`;
}

export function getDownloadUrl(filename: string): string {
  return `${API_BASE}/api/download/${filename}`;
}
