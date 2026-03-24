"use client";

import React, { useState, useCallback } from "react";
import {
  Mic2,
  AudioWaveform,
  FileAudio,
  CheckCircle2,
  ChevronRight,
  Cpu,
} from "lucide-react";
import AudioUploader from "@/components/audio-uploader";
import WaveformEditor from "@/components/waveform-editor";
import TextInput from "@/components/text-input";
import SynthesisPanel from "@/components/synthesis-panel";
import { UploadResult, TrimResult, getUploadedAudioUrl } from "@/lib/api";
import { formatDuration, formatFileSize } from "@/lib/utils";

export default function Home() {
  // State
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [trimResult, setTrimResult] = useState<TrimResult | null>(null);
  const [text, setText] = useState("");
  const [refText, setRefText] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  const handleUploadComplete = useCallback((result: UploadResult) => {
    setUploadResult(result);
    setTrimResult(null); // Reset trim on new upload
  }, []);

  const handleTrimComplete = useCallback((result: TrimResult) => {
    setTrimResult(result);
  }, []);

  const handleReset = useCallback(() => {
    setUploadResult(null);
    setTrimResult(null);
    setText("");
    setRefText("");
  }, []);

  // Step tracking
  const currentStep = !uploadResult ? 1 : !trimResult ? 2 : 3;

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background decorations */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent/5 rounded-full blur-[120px]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/3 rounded-full blur-[200px]" />
      </div>

      {/* Header */}
      <header className="relative border-b border-border/50 bg-card/30 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent shadow-lg">
                <Mic2 className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold gradient-text">
                  VN-VoiceClone Pro
                </h1>
                <p className="text-xs text-muted-foreground">
                  Clone giọng nói tiếng Việt với AI • Chạy offline
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="hidden sm:flex items-center gap-2 text-xs text-muted-foreground bg-secondary/50 rounded-full px-3 py-1.5">
                <Cpu className="w-3 h-3" />
                <span>VieNeu-TTS</span>
              </div>
              {uploadResult && (
                <button
                  onClick={handleReset}
                  className="text-xs text-muted-foreground hover:text-foreground transition-colors px-3 py-1.5 rounded-full hover:bg-secondary/50"
                >
                  Bắt đầu lại
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Step Indicator */}
      <div className="relative max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center gap-2 text-sm">
          {[
            { n: 1, label: "Tải audio" },
            { n: 2, label: "Chọn mẫu giọng" },
            { n: 3, label: "Tổng hợp" },
          ].map((step, i) => (
            <React.Fragment key={step.n}>
              {i > 0 && (
                <ChevronRight className="w-4 h-4 text-muted-foreground/30" />
              )}
              <div
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-all duration-300 ${
                  step.n === currentStep
                    ? "bg-primary/15 text-primary font-medium"
                    : step.n < currentStep
                    ? "text-green-400"
                    : "text-muted-foreground/50"
                }`}
              >
                {step.n < currentStep ? (
                  <CheckCircle2 className="w-4 h-4" />
                ) : (
                  <span
                    className={`flex items-center justify-center w-5 h-5 rounded-full text-xs font-bold ${
                      step.n === currentStep
                        ? "bg-primary text-white"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {step.n}
                  </span>
                )}
                <span className="hidden sm:inline">{step.label}</span>
              </div>
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <main className="relative max-w-7xl mx-auto px-6 pb-12">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* ── Left Panel: Audio Upload + Waveform (3/5) ── */}
          <div className="lg:col-span-3 space-y-6">
            {/* Step 1: Upload */}
            <section className="glass-card rounded-2xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
                  <FileAudio className="w-4 h-4 text-primary" />
                </div>
                <div>
                  <h2 className="text-base font-semibold text-foreground">
                    Bước 1: Tải lên file audio
                  </h2>
                  <p className="text-xs text-muted-foreground">
                    Tải lên audio chứa giọng nói bạn muốn clone
                  </p>
                </div>
              </div>

              {!uploadResult ? (
                <AudioUploader
                  onUploadComplete={handleUploadComplete}
                  isUploading={isUploading}
                  setIsUploading={setIsUploading}
                />
              ) : (
                <div className="flex items-center gap-3 p-3 bg-green-500/5 border border-green-500/20 rounded-xl text-sm animate-fade-in">
                  <CheckCircle2 className="w-5 h-5 text-green-400 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-foreground truncate">
                      {uploadResult.original_name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatDuration(uploadResult.duration)} •{" "}
                      {formatFileSize(uploadResult.size)} •{" "}
                      {uploadResult.sample_rate}Hz
                    </p>
                  </div>
                  <button
                    onClick={handleReset}
                    className="text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded-lg hover:bg-secondary/50 transition-colors"
                  >
                    Đổi file
                  </button>
                </div>
              )}
            </section>

            {/* Step 2: Waveform Editor */}
            {uploadResult && (
              <section className="glass-card rounded-2xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
                    <AudioWaveform className="w-4 h-4 text-primary" />
                  </div>
                  <div>
                    <h2 className="text-base font-semibold text-foreground">
                      Bước 2: Chọn đoạn mẫu giọng
                    </h2>
                    <p className="text-xs text-muted-foreground">
                      Kéo thả vùng chọn trên waveform (khuyến nghị 5-10 giây)
                    </p>
                  </div>
                </div>

                <WaveformEditor
                  audioUrl={getUploadedAudioUrl(uploadResult.filename)}
                  filename={uploadResult.filename}
                  totalDuration={uploadResult.duration}
                  onTrimComplete={handleTrimComplete}
                />

                {/* Trim success info */}
                {trimResult && (
                  <div className="mt-4 flex items-center gap-3 p-3 bg-green-500/5 border border-green-500/20 rounded-xl text-sm animate-slide-up">
                    <CheckCircle2 className="w-5 h-5 text-green-400 shrink-0" />
                    <div>
                      <p className="font-medium text-foreground">
                        Đã cắt mẫu giọng thành công
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Độ dài: {trimResult.duration}s •{" "}
                        {formatDuration(trimResult.start)} →{" "}
                        {formatDuration(trimResult.end)}
                      </p>
                    </div>
                  </div>
                )}
              </section>
            )}
          </div>

          {/* ── Right Panel: Text + Synthesis (2/5) ── */}
          <div className="lg:col-span-2 space-y-6">
            {/* Step 3: Text Input + Synthesis */}
            <section className="glass-card rounded-2xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent/10">
                  <Mic2 className="w-4 h-4 text-accent" />
                </div>
                <div>
                  <h2 className="text-base font-semibold text-foreground">
                    Bước 3: Tổng hợp giọng nói
                  </h2>
                  <p className="text-xs text-muted-foreground">
                    Nhập văn bản và tạo giọng nói AI
                  </p>
                </div>
              </div>

              <div className="space-y-6">
                <TextInput
                  text={text}
                  setText={setText}
                  refText={refText}
                  setRefText={setRefText}
                  disabled={!trimResult}
                />

                <div className="border-t border-border/50 pt-4">
                  <SynthesisPanel
                    trimmedFilename={trimResult?.trimmed_filename || null}
                    text={text}
                    refText={refText}
                  />
                </div>
              </div>
            </section>

            {/* Info Card */}
            <div className="glass-card rounded-2xl p-5 text-xs text-muted-foreground space-y-3">
              <h3 className="text-sm font-medium text-foreground">
                💡 Mẹo sử dụng
              </h3>
              <ul className="space-y-2 list-none">
                <li className="flex gap-2">
                  <span className="text-primary">•</span>
                  <span>
                    Chọn đoạn audio rõ ràng, ít tạp âm để đạt chất lượng
                    clone tốt nhất
                  </span>
                </li>
                <li className="flex gap-2">
                  <span className="text-primary">•</span>
                  <span>
                    Độ dài mẫu lý tưởng là 5-10 giây, tối thiểu 3 giây
                  </span>
                </li>
                <li className="flex gap-2">
                  <span className="text-primary">•</span>
                  <span>
                    Thêm văn bản tham chiếu (nội dung nói trong mẫu) giúp
                    cải thiện chất lượng đáng kể
                  </span>
                </li>
                <li className="flex gap-2">
                  <span className="text-primary">•</span>
                  <span>
                    Lần chạy đầu tiên sẽ tải model (~1.5GB), sau đó sẽ nhanh
                    hơn
                  </span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative border-t border-border/30 bg-card/20 backdrop-blur-xl mt-8">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between text-xs text-muted-foreground">
          <span>VN-VoiceClone Pro v1.0 • Powered by VieNeu-TTS</span>
          <span>Chạy hoàn toàn offline trên máy của bạn</span>
        </div>
      </footer>
    </div>
  );
}
