"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";
import {
  Download,
  Play,
  Pause,
  Clock,
  FileText,
  Sparkles,
  CheckCircle2,
  Gauge,
  Zap,
  Crown,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  synthesizeVoice,
  getSynthesisProgress,
  SynthesisResult,
  getOutputAudioUrl,
  getDownloadUrl,
} from "@/lib/api";
import { formatDuration } from "@/lib/utils";

interface SynthesisPanelProps {
  trimmedFilename: string | null;
  text: string;
  refText: string;
  voiceProfileId: string | null;
}

type Quality = "fast" | "standard" | "high";

const QUALITY_OPTIONS: { value: Quality; label: string; icon: React.ReactNode; desc: string }[] = [
  { value: "fast", label: "Nhanh", icon: <Zap className="w-3.5 h-3.5" />, desc: "~2× nhanh hơn" },
  { value: "standard", label: "Chuẩn", icon: <Gauge className="w-3.5 h-3.5" />, desc: "Cân bằng" },
  { value: "high", label: "Cao cấp", icon: <Crown className="w-3.5 h-3.5" />, desc: "Chất lượng tốt nhất" },
];

export default function SynthesisPanel({
  trimmedFilename,
  text,
  refText,
  voiceProfileId,
}: SynthesisPanelProps) {
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [result, setResult] = useState<SynthesisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioCurrentTime, setAudioCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [engine, setEngine] = useState<"vieneu" | "f5-tts">("f5-tts");
  const [speed, setSpeed] = useState(1.0);
  const [quality, setQuality] = useState<Quality>("standard");
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const audioRef = useRef<HTMLAudioElement>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const canSynthesize =
    (trimmedFilename || voiceProfileId) && text.trim().length > 0;

  // Cleanup progress polling on unmount
  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, []);

  const startProgressPolling = useCallback((taskId: string) => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }
    progressIntervalRef.current = setInterval(async () => {
      try {
        const p = await getSynthesisProgress(taskId);
        setProgress(p.progress);
        setProgressMessage(p.message);
        if (p.status === "done" || p.status === "error") {
          if (progressIntervalRef.current) {
            clearInterval(progressIntervalRef.current);
            progressIntervalRef.current = null;
          }
        }
      } catch {
        // Ignore poll errors
      }
    }, 1000);
  }, []);

  const handleSynthesize = async () => {
    if (!text.trim()) return;
    if (!trimmedFilename && !voiceProfileId) return;

    setIsSynthesizing(true);
    setError(null);
    setResult(null);
    setProgress(0);
    setProgressMessage("Đang bắt đầu...");

    try {
      const res = await synthesizeVoice(text, {
        trimmedFilename: trimmedFilename || undefined,
        refText: refText || undefined,
        voiceProfileId: voiceProfileId || undefined,
        engine,
        speed: engine === "f5-tts" ? speed : undefined,
        quality: engine === "f5-tts" ? quality : undefined,
      });

      // Start polling progress if we have task_id
      if (res.task_id) {
        startProgressPolling(res.task_id);
      }

      setResult(res);
      setProgress(100);
      setProgressMessage("Hoàn tất!");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Lỗi tổng hợp giọng nói");
    } finally {
      setIsSynthesizing(false);
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
    }
  };

  const togglePlayOutput = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
  };

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onEnded = () => setIsPlaying(false);
    const onTimeUpdate = () => setAudioCurrentTime(audio.currentTime);
    const onLoadedMetadata = () => setAudioDuration(audio.duration);

    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);
    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("loadedmetadata", onLoadedMetadata);

    return () => {
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
    };
  }, [result]);

  return (
    <div className="space-y-4">
      {/* Engine Toggle */}
      <div className="flex items-center gap-2 p-1 rounded-xl bg-secondary/50 border border-border/50">
        <button
          type="button"
          onClick={() => setEngine("f5-tts")}
          className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200 ${
            engine === "f5-tts"
              ? "bg-gradient-to-r from-primary to-accent text-white shadow-lg shadow-primary/25"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          🚀 F5-TTS — Chất lượng cao
        </button>
        <button
          type="button"
          onClick={() => setEngine("vieneu")}
          className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200 ${
            engine === "vieneu"
              ? "bg-gradient-to-r from-primary to-accent text-white shadow-lg shadow-primary/25"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          ⚡ VieNeu — Nhanh
        </button>
      </div>

      {/* F5-TTS Options */}
      {engine === "f5-tts" && (
        <div className="space-y-3 p-3 rounded-xl bg-secondary/30 border border-border/30">
          {/* Quality Selector */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">Chất lượng</label>
            <div className="flex gap-1.5">
              {QUALITY_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => setQuality(opt.value)}
                  className={`flex-1 flex flex-col items-center gap-1 px-2 py-2 rounded-lg text-xs transition-all duration-200 ${
                    quality === opt.value
                      ? "bg-gradient-to-br from-primary/20 to-accent/20 border border-primary/40 text-foreground shadow-sm"
                      : "border border-transparent text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                  }`}
                >
                  <span className="flex items-center gap-1 font-medium">
                    {opt.icon} {opt.label}
                  </span>
                  <span className="text-[10px] opacity-70">{opt.desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Speed Slider */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-muted-foreground">Tốc độ đọc</label>
              <span className="text-xs font-mono text-primary font-semibold">{speed.toFixed(1)}×</span>
            </div>
            <div className="relative">
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="w-full h-2 rounded-full appearance-none cursor-pointer
                  bg-gradient-to-r from-blue-500/30 via-primary/30 to-orange-500/30
                  [&::-webkit-slider-thumb]:appearance-none
                  [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
                  [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-primary
                  [&::-webkit-slider-thumb]:shadow-lg [&::-webkit-slider-thumb]:shadow-primary/30
                  [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white/20
                  [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:duration-150
                  [&::-webkit-slider-thumb]:hover:scale-125"
              />
              <div className="flex justify-between text-[10px] text-muted-foreground mt-1 px-0.5">
                <span>Chậm</span>
                <span>Bình thường</span>
                <span>Nhanh</span>
              </div>
            </div>
          </div>

          <p className="text-xs text-emerald-400">
            ✨ Tự động: chuẩn hóa số/đơn vị, xóa khoảng lặng, kiểm tra Whisper
          </p>
        </div>
      )}

      {/* Synthesize Button */}
      <Button
        variant="glow"
        size="lg"
        className="w-full gap-3 text-base h-14"
        onClick={handleSynthesize}
        disabled={!canSynthesize || isSynthesizing}
      >
        {isSynthesizing ? (
          <>
            <div className="relative flex items-center justify-center w-5 h-5">
              <div className="absolute w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            </div>
            <span>Đang tổng hợp giọng nói...</span>
          </>
        ) : (
          <>
            <Sparkles className="w-5 h-5" />
            <span>Tổng hợp giọng nói</span>
          </>
        )}
      </Button>

      {/* Progress Display */}
      {isSynthesizing && (
        <div className="space-y-2 animate-slide-up">
          <div className="relative w-full h-2.5 bg-secondary rounded-full overflow-hidden">
            <div
              className="absolute top-0 left-0 h-full bg-gradient-to-r from-primary via-accent to-primary rounded-full transition-all duration-500 ease-out"
              style={{ width: `${Math.max(progress, 5)}%` }}
            />
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-pulse" />
          </div>
          <div className="flex justify-between items-center">
            <p className="text-xs text-muted-foreground">{progressMessage}</p>
            <span className="text-xs font-mono text-primary">{progress}%</span>
          </div>
        </div>
      )}

      {/* Mode indicator */}
      {canSynthesize && !isSynthesizing && !result && (
        <div className="text-xs text-muted-foreground px-1">
          {voiceProfileId ? (
            <p className="flex items-center gap-2 text-primary">
              <CheckCircle2 className="w-3 h-3" />
              Sử dụng giọng mẫu đã lưu (nhanh & ổn định hơn)
            </p>
          ) : (
            <p className="flex items-center gap-2 text-yellow-400">
              ○ Dùng audio thô — tạo giọng mẫu để ổn định hơn
            </p>
          )}
        </div>
      )}

      {/* Requirements hint */}
      {!canSynthesize && !isSynthesizing && !result && (
        <div className="text-xs text-muted-foreground space-y-1 px-1">
          <p className="flex items-center gap-2">
            <span
              className={
                trimmedFilename || voiceProfileId
                  ? "text-green-400"
                  : "text-yellow-400"
              }
            >
              {trimmedFilename || voiceProfileId ? "✓" : "○"}
            </span>
            {trimmedFilename || voiceProfileId
              ? "Đã có giọng mẫu"
              : "Cần cắt mẫu giọng hoặc chọn giọng đã lưu"}
          </p>
          <p className="flex items-center gap-2">
            <span
              className={text.trim() ? "text-green-400" : "text-yellow-400"}
            >
              {text.trim() ? "✓" : "○"}
            </span>
            {text.trim()
              ? "Đã nhập văn bản"
              : "Cần nhập văn bản tiếng Việt"}
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-4 rounded-xl bg-destructive/10 border border-destructive/20 text-sm text-destructive animate-slide-up">
          <p className="font-medium mb-1">Lỗi tổng hợp</p>
          <p className="text-xs opacity-80">{error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-3 animate-slide-up">
          {/* Success Header */}
          <div className="flex items-center gap-2 text-green-400 text-sm">
            <CheckCircle2 className="w-4 h-4" />
            <span className="font-medium">Tổng hợp thành công!</span>
            {result.coverage !== undefined && result.coverage > 0 && (
              <span className="text-xs text-muted-foreground ml-auto">
                Độ phủ: {Math.round(result.coverage * 100)}%
              </span>
            )}
          </div>

          {/* Audio Player */}
          <div className="relative rounded-2xl bg-gradient-to-br from-primary/10 via-secondary/50 to-accent/10 border border-primary/20 p-5 overflow-hidden">
            {/* Decorative glow */}
            <div className="absolute -top-12 -right-12 w-24 h-24 bg-primary/20 rounded-full blur-3xl" />
            <div className="absolute -bottom-8 -left-8 w-20 h-20 bg-accent/20 rounded-full blur-3xl" />

            <div className="relative space-y-4">
              {/* Play + Progress */}
              <div className="flex items-center gap-4">
                <Button
                  variant="glow"
                  size="icon"
                  className="w-12 h-12 rounded-full shrink-0"
                  onClick={togglePlayOutput}
                >
                  {isPlaying ? (
                    <Pause className="w-5 h-5" />
                  ) : (
                    <Play className="w-5 h-5 ml-0.5" />
                  )}
                </Button>

                <div className="flex-1 space-y-1">
                  {/* Progress bar */}
                  <div
                    className="relative w-full h-2 bg-secondary rounded-full overflow-hidden cursor-pointer"
                    onClick={(e) => {
                      if (!audioRef.current || !audioDuration) return;
                      const rect = e.currentTarget.getBoundingClientRect();
                      const ratio = (e.clientX - rect.left) / rect.width;
                      audioRef.current.currentTime = ratio * audioDuration;
                    }}
                  >
                    <div
                      className="absolute top-0 left-0 h-full bg-gradient-to-r from-primary to-accent rounded-full transition-all duration-150"
                      style={{
                        width: `${
                          audioDuration
                            ? (audioCurrentTime / audioDuration) * 100
                            : 0
                        }%`,
                      }}
                    />
                  </div>

                  {/* Time */}
                  <div className="flex justify-between text-xs font-mono text-muted-foreground">
                    <span>{formatDuration(audioCurrentTime)}</span>
                    <span>{formatDuration(audioDuration)}</span>
                  </div>
                </div>
              </div>

              {/* Info row */}
              <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  Xử lý: {result.processing_time}s
                </span>
                <span className="flex items-center gap-1">
                  <FileText className="w-3 h-3" />
                  {result.text_length} ký tự
                </span>
                {result.quality && (
                  <span className="text-primary/70">
                    {result.quality === "high" ? "💎" : result.quality === "fast" ? "⚡" : "🎯"} {result.quality}
                  </span>
                )}
                {result.speed && result.speed !== 1.0 && (
                  <span className="text-primary/70">
                    {result.speed}×
                  </span>
                )}
              </div>
            </div>

            <audio
              ref={audioRef}
              src={getOutputAudioUrl(result.output_filename)}
              preload="auto"
            />
          </div>

          {/* Download Button */}
          <a
            href={getDownloadUrl(result.output_filename)}
            download
            className="block"
          >
            <Button variant="outline" className="w-full gap-2">
              <Download className="w-4 h-4" />
              Tải xuống file WAV
            </Button>
          </a>
        </div>
      )}
    </div>
  );
}
