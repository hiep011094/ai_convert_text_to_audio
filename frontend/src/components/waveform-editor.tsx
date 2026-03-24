"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, { Region } from "wavesurfer.js/dist/plugins/regions.js";
import {
  Play,
  Pause,
  SkipBack,
  Scissors,
  Volume2,
  ZoomIn,
  ZoomOut,

} from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatDuration } from "@/lib/utils";
import { trimAudio, TrimResult } from "@/lib/api";

interface WaveformEditorProps {
  audioUrl: string;
  filename: string;
  totalDuration: number;
  onTrimComplete: (result: TrimResult) => void;
}

export default function WaveformEditor({
  audioUrl,
  filename,
  totalDuration,
  onTrimComplete,
}: WaveformEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const activeRegionRef = useRef<Region | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrTime] = useState(0);
  const [duration, setDuration] = useState(totalDuration);
  const [regionStart, setRegionStart] = useState(0);
  const [regionEnd, setRegionEnd] = useState(Math.min(7, totalDuration));
  const [zoom, setZoom] = useState(50);
  const [isTrimming, setIsTrimming] = useState(false);
  const [isReady, setIsReady] = useState(false);

  // Initialize WaveSurfer
  useEffect(() => {
    if (!containerRef.current) return;

    let destroyed = false;
    const regions = RegionsPlugin.create();
    regionsRef.current = regions;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "hsl(262 60% 45%)",
      progressColor: "hsl(262 83% 65%)",
      cursorColor: "hsl(192 91% 55%)",
      cursorWidth: 2,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      height: 120,
      normalize: true,
      plugins: [regions],
    });

    wavesurferRef.current = ws;

    // Suppress AbortError thrown when destroy() is called during loading
    ws.on("error", (err: Error) => {
      if (err.name === "AbortError" || destroyed) return;
      console.error("WaveSurfer error:", err);
    });

    ws.load(audioUrl).catch((err: Error) => {
      if (err.name === "AbortError" || destroyed) return;
      console.error("WaveSurfer load error:", err);
    });

    ws.on("ready", () => {
      if (destroyed) return;
      setDuration(ws.getDuration());
      setIsReady(true);

      // Create initial selection region
      const end = Math.min(7, ws.getDuration());
      const region = regions.addRegion({
        start: 0,
        end: end,
        color: "rgba(139, 92, 246, 0.15)",
        drag: true,
        resize: true,
      });
      activeRegionRef.current = region;
      setRegionStart(0);
      setRegionEnd(end);
    });

    ws.on("timeupdate", (time: number) => {
      if (!destroyed) setCurrTime(time);
    });

    ws.on("play", () => !destroyed && setIsPlaying(true));
    ws.on("pause", () => !destroyed && setIsPlaying(false));

    regions.on("region-updated", (region: Region) => {
      if (destroyed) return;
      setRegionStart(region.start);
      setRegionEnd(region.end);
      activeRegionRef.current = region;
    });

    return () => {
      destroyed = true;
      ws.destroy();
    };
  }, [audioUrl]);

  // Update zoom
  useEffect(() => {
    if (isReady) {
      wavesurferRef.current?.zoom(zoom);
    }
  }, [zoom, isReady]);

  const togglePlay = useCallback(() => {
    wavesurferRef.current?.playPause();
  }, []);

  const playRegion = useCallback(() => {
    if (activeRegionRef.current) {
      activeRegionRef.current.play();
    }
  }, []);

  const resetToStart = useCallback(() => {
    wavesurferRef.current?.seekTo(0);
  }, []);

  const handleTrim = useCallback(async () => {
    if (!activeRegionRef.current) return;

    const start = activeRegionRef.current.start;
    const end = activeRegionRef.current.end;
    const len = end - start;

    if (len < 1) {
      alert("Vui lòng chọn đoạn dài ít nhất 1 giây");
      return;
    }
    if (len > 30) {
      alert("Đoạn chọn không được quá 30 giây. Vui lòng thu hẹp vùng chọn.");
      return;
    }

    setIsTrimming(true);
    try {
      const result = await trimAudio(filename, start, end);
      onTrimComplete(result);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : "Lỗi khi cắt audio");
    } finally {
      setIsTrimming(false);
    }
  }, [filename, onTrimComplete]);

  const regionDuration = regionEnd - regionStart;

  return (
    <div className="space-y-4 animate-slide-up">
      {/* Waveform Container */}
      <div className="relative rounded-2xl bg-secondary/30 border border-border p-4 overflow-hidden">
        {!isReady && (
          <div className="absolute inset-0 flex items-center justify-center bg-card/80 z-10 rounded-2xl">
            <div className="flex items-center gap-3 text-muted-foreground">
              <div className="w-5 h-5 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
              <span className="text-sm">Đang tải waveform...</span>
            </div>
          </div>
        )}
        <div ref={containerRef} className="w-full" />
      </div>

      {/* Controls Row */}
      <div className="flex items-center gap-2 flex-wrap">
        {/* Playback Controls */}
        <div className="flex items-center gap-1 bg-secondary/50 rounded-xl p-1">
          <Button variant="ghost" size="icon" onClick={resetToStart} title="Về đầu">
            <SkipBack className="w-4 h-4" />
          </Button>
          <Button variant="ghost" size="icon" onClick={togglePlay} title={isPlaying ? "Tạm dừng" : "Phát"}>
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </Button>
          <Button variant="ghost" size="icon" onClick={playRegion} title="Phát vùng chọn">
            <Volume2 className="w-4 h-4" />
          </Button>
        </div>

        {/* Time Display */}
        <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground bg-secondary/50 rounded-xl px-3 py-2">
          <span>{formatDuration(currentTime)}</span>
          <span>/</span>
          <span>{formatDuration(duration)}</span>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-1 bg-secondary/50 rounded-xl p-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setZoom(Math.max(10, zoom - 20))}
            title="Thu nhỏ"
          >
            <ZoomOut className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setZoom(Math.min(300, zoom + 20))}
            title="Phóng to"
          >
            <ZoomIn className="w-4 h-4" />
          </Button>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Trim Button */}
        <Button
          variant="glow"
          size="default"
          onClick={handleTrim}
          disabled={isTrimming || !isReady}
          className="gap-2"
        >
          {isTrimming ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Đang cắt...
            </>
          ) : (
            <>
              <Scissors className="w-4 h-4" />
              Cắt mẫu giọng
            </>
          )}
        </Button>
      </div>

      {/* Region Info */}
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2 bg-primary/10 text-primary rounded-xl px-3 py-2">
          <span className="text-xs text-primary/70">Bắt đầu:</span>
          <span className="font-mono font-medium">{formatDuration(regionStart)}</span>
        </div>
        <div className="flex items-center gap-2 bg-primary/10 text-primary rounded-xl px-3 py-2">
          <span className="text-xs text-primary/70">Kết thúc:</span>
          <span className="font-mono font-medium">{formatDuration(regionEnd)}</span>
        </div>
        <div
          className={`flex items-center gap-2 rounded-xl px-3 py-2 ${
            regionDuration >= 3 && regionDuration <= 10
              ? "bg-green-500/10 text-green-400"
              : "bg-yellow-500/10 text-yellow-400"
          }`}
        >
          <span className="text-xs opacity-70">Độ dài:</span>
          <span className="font-mono font-medium">{regionDuration.toFixed(1)}s</span>
        </div>
        {(regionDuration < 3 || regionDuration > 10) && (
          <span className="text-xs text-yellow-400">
            ⚠ Nên chọn 3-10 giây để clone giọng tốt nhất
          </span>
        )}
      </div>
    </div>
  );
}
