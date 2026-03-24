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
}

export default function SynthesisPanel({
  trimmedFilename,
  text,
  refText,
}: SynthesisPanelProps) {
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [result, setResult] = useState<SynthesisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioCurrentTime, setAudioCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);

  const canSynthesize = trimmedFilename && text.trim().length > 0;

  const handleSynthesize = async () => {
    if (!trimmedFilename || !text.trim()) return;

    setIsSynthesizing(true);
    setError(null);
    setResult(null);

    try {
      const res = await synthesizeVoice(trimmedFilename, text, refText);
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

      {/* Requirements hint */}
      {!canSynthesize && !isSynthesizing && !result && (
        <div className="text-xs text-muted-foreground space-y-1 px-1">
          <p className="flex items-center gap-2">
            <span className={trimmedFilename ? "text-green-400" : "text-yellow-400"}>
              {trimmedFilename ? "✓" : "○"}
            </span>
            {trimmedFilename
              ? "Đã cắt mẫu giọng"
              : "Cần cắt mẫu giọng từ audio"}
          </p>
          <p className="flex items-center gap-2">
            <span className={text.trim() ? "text-green-400" : "text-yellow-400"}>
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
