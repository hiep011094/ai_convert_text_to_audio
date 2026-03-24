"use client";

import React, { useCallback, useRef, useState } from "react";
import { Upload, X, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { uploadAudio, UploadResult } from "@/lib/api";

interface AudioUploaderProps {
  onUploadComplete: (result: UploadResult) => void;
  isUploading: boolean;
  setIsUploading: (v: boolean) => void;
}

export default function AudioUploader({
  onUploadComplete,
  isUploading,
  setIsUploading,
}: AudioUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);

      const validTypes = [
        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/flac",
        "audio/ogg",
        "audio/mp4",
        "audio/x-m4a",
      ];
      const ext = file.name.split(".").pop()?.toLowerCase();
      const validExts = ["mp3", "wav", "flac", "ogg", "m4a"];

      if (!validTypes.includes(file.type) && !validExts.includes(ext || "")) {
        setError("Định dạng không hỗ trợ. Vui lòng chọn file MP3, WAV, FLAC, OGG hoặc M4A.");
        return;
      }

      if (file.size > 100 * 1024 * 1024) {
        setError("File quá lớn. Kích thước tối đa là 100MB.");
        return;
      }

      setIsUploading(true);
      try {
        const result = await uploadAudio(file);
        onUploadComplete(result);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Lỗi khi upload file");
      } finally {
        setIsUploading(false);
      }
    },
    [onUploadComplete, setIsUploading]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="space-y-3">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={cn(
          "relative flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed p-8 cursor-pointer transition-all duration-300",
          isDragging
            ? "dropzone-active border-primary bg-primary/5 scale-[1.01]"
            : "border-border hover:border-primary/50 hover:bg-secondary/30",
          isUploading && "pointer-events-none opacity-60"
        )}
      >
        {isUploading ? (
          <>
            <div className="relative">
              <div className="w-16 h-16 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
            </div>
            <p className="text-sm text-muted-foreground">Đang tải lên...</p>
          </>
        ) : (
          <>
            <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 text-primary">
              <Upload className="w-7 h-7" />
            </div>
            <div className="text-center">
              <p className="text-base font-medium text-foreground">
                Kéo thả file audio vào đây
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                hoặc <span className="text-primary font-medium">nhấp để chọn file</span>
              </p>
              <p className="text-xs text-muted-foreground mt-2">
                Hỗ trợ: MP3, WAV, FLAC, OGG, M4A • Tối đa 100MB
              </p>
            </div>
          </>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a"
          className="hidden"
          onChange={handleInputChange}
        />
      </div>

      {error && (
        <div className="flex items-center gap-2 p-3 rounded-xl bg-destructive/10 border border-destructive/20 text-sm animate-slide-up">
          <AlertCircle className="w-4 h-4 text-destructive shrink-0" />
          <span className="text-destructive">{error}</span>
          <button
            onClick={() => setError(null)}
            className="ml-auto text-destructive/60 hover:text-destructive"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
