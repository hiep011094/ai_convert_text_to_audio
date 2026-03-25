"use client";

import React, { useRef, useState, useEffect } from "react";
import {
  Download,
  Play,
  Pause,
  Clock,
  FileText,
  Sparkles,
  CheckCircle2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  synthesizeVoice,
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
  const audioRef = useRef<HTMLAudioElement>(null);

  const canSynthesize =
    (trimmedFilename || voiceProfileId) && text.trim().length > 0;

  const handleSynthesize = async () => {
    if (!text.trim()) return;
    if (!trimmedFilename && !voiceProfileId) return;

    setIsSynthesizing(true);
    setError(null);
    setResult(null);

    try {
      const res = await synthesizeVoice(text, {
        trimmedFilename: trimmedFilename || undefined,
        refText: refText || undefined,
        voiceProfileId: voiceProfileId || undefined,
        engine,
      });
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Lỗi tổng hợp giọng nói");
    } finally {
      setIsSynthesizing(false);
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

      {engine === "f5-tts" && (
        <p className="text-xs text-emerald-400 px-1">✨ F5-TTS: Voice cloning chất lượng cao, đọc đầy đủ 100% văn bản</p>
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
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  Xử lý: {result.processing_time}s
                </span>
                <span className="flex items-center gap-1">
                  <FileText className="w-3 h-3" />
                  {result.text_length} ký tự
                </span>
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
