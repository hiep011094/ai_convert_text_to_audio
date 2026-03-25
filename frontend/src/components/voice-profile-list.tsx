"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  User,
  Trash2,
  Play,
  Pause,
  Plus,
  Dna,
  Loader2,
  CheckCircle2,
  Shield,
  Cpu,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  VoiceProfile,
  createVoiceProfile,
  listVoiceProfiles,
  deleteVoiceProfile,
  getProfileAudioUrl,
  getSynthesisProgress,
} from "@/lib/api";

interface VoiceProfileListProps {
  trimmedFilename: string | null;
  refText: string;
  selectedProfileId: string | null;
  onSelectProfile: (profileId: string | null) => void;
  onProfileCreated?: () => void;
}

export default function VoiceProfileList({
  trimmedFilename,
  refText,
  selectedProfileId,
  onSelectProfile,
  onProfileCreated,
}: VoiceProfileListProps) {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [profileName, setProfileName] = useState("");
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [profileEngine, setProfileEngine] = useState<"vieneu" | "f5-tts">("f5-tts");
  const [createProgress, setCreateProgress] = useState(0);
  const [createMessage, setCreateMessage] = useState("");
  const audioRef = useRef<HTMLAudioElement>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Load profiles on mount
  useEffect(() => {
    loadProfiles();
  }, []);

  // Cleanup progress polling on unmount
  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, []);

  const loadProfiles = async () => {
    try {
      const data = await listVoiceProfiles();
      setProfiles(data);
    } catch {
      console.error("Failed to load profiles");
    }
  };

  const startProgressPolling = useCallback((taskId: string) => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }
    progressIntervalRef.current = setInterval(async () => {
      try {
        const p = await getSynthesisProgress(taskId);
        setCreateProgress(p.progress);
        setCreateMessage(p.message);
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

  const handleCreate = async () => {
    if (!trimmedFilename || !profileName.trim()) return;

    setIsCreating(true);
    setError(null);
    setCreateProgress(0);
    setCreateMessage("Đang bắt đầu...");

    try {
      const newProfile = await createVoiceProfile(
        trimmedFilename,
        profileName.trim(),
        refText || undefined,
        profileEngine
      );

      // Start progress polling if task_id available
      if (newProfile.task_id) {
        startProgressPolling(newProfile.task_id);
      }

      setProfiles((prev) => [newProfile, ...prev]);
      setProfileName("");
      setShowCreateForm(false);
      onSelectProfile(newProfile.id);
      onProfileCreated?.();
      setCreateProgress(100);
      setCreateMessage("Hoàn tất!");
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "Lỗi tạo voice profile"
      );
    } finally {
      setIsCreating(false);
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteVoiceProfile(id);
      setProfiles((prev) => prev.filter((p) => p.id !== id));
      if (selectedProfileId === id) {
        onSelectProfile(null);
      }
    } catch (err: unknown) {
      console.error("Delete failed:", err);
    }
  };

  const togglePlay = (profile: VoiceProfile) => {
    if (playingId === profile.id) {
      audioRef.current?.pause();
      setPlayingId(null);
    } else {
      if (audioRef.current) {
        audioRef.current.src = getProfileAudioUrl(profile.calibration_audio);
        audioRef.current.play();
        setPlayingId(profile.id);
      }
    }
  };

  const formatDate = (iso: string) => {
    try {
      const d = new Date(iso);
      return d.toLocaleDateString("vi-VN", {
        day: "2-digit",
        month: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "";
    }
  };

  const getQualityBadge = (score?: number) => {
    if (score === undefined || score < 0) return null;
    if (score >= 0.8) return { label: "Tốt", color: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20" };
    if (score >= 0.5) return { label: "Khá", color: "text-yellow-400 bg-yellow-400/10 border-yellow-400/20" };
    return { label: "Thấp", color: "text-orange-400 bg-orange-400/10 border-orange-400/20" };
  };

  return (
    <>
      <div className="space-y-3">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
            <Dna className="w-4 h-4 text-primary" />
            Giọng mẫu đã lưu
            {profiles.length > 0 && (
              <span className="text-xs text-muted-foreground bg-secondary rounded-full px-2 py-0.5">
                {profiles.length}
              </span>
            )}
          </h3>
          {trimmedFilename && !showCreateForm && (
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5 text-xs h-7"
              onClick={() => setShowCreateForm(true)}
            >
              <Plus className="w-3 h-3" />
              Tạo mới
            </Button>
          )}
        </div>

        {/* Create Form */}
        {showCreateForm && (
          <div className="p-3 rounded-xl bg-primary/5 border border-primary/20 space-y-3 animate-slide-up">
            <input
              type="text"
              placeholder="Tên giọng mẫu (VD: Giọng nam Hà Nội)"
              value={profileName}
              onChange={(e) => setProfileName(e.target.value)}
              className="w-full px-3 py-2 text-sm rounded-lg bg-secondary/50 border border-border/50 focus:border-primary/50 focus:outline-none transition-colors"
              autoFocus
            />

            {/* Engine Toggle */}
            <div className="flex items-center gap-1.5 p-1 rounded-lg bg-secondary/30 border border-border/30">
              <button
                type="button"
                onClick={() => setProfileEngine("f5-tts")}
                className={`flex-1 px-2 py-1.5 rounded-md text-xs font-medium transition-all duration-200 ${
                  profileEngine === "f5-tts"
                    ? "bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                🚀 F5-TTS
              </button>
              <button
                type="button"
                onClick={() => setProfileEngine("vieneu")}
                className={`flex-1 px-2 py-1.5 rounded-md text-xs font-medium transition-all duration-200 ${
                  profileEngine === "vieneu"
                    ? "bg-gradient-to-r from-violet-500 to-purple-500 text-white shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                ⚡ VieNeu
              </button>
            </div>
            {profileEngine === "f5-tts" && (
              <p className="text-xs text-emerald-400">✨ Chất lượng cao, tự kiểm tra Whisper (~30-60s)</p>
            )}
            {profileEngine === "vieneu" && (
              <p className="text-xs text-violet-400">⚡ Nhanh (~10-20s), phù hợp cho test nhanh</p>
            )}

            <div className="flex gap-2">
              <Button
                variant="glow"
                size="sm"
                className="gap-1.5 flex-1"
                onClick={handleCreate}
                disabled={isCreating || !profileName.trim()}
              >
                {isCreating ? (
                  <>
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Đang tạo...
                  </>
                ) : (
                  <>
                    <Dna className="w-3 h-3" />
                    Tạo giọng mẫu
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setShowCreateForm(false);
                  setError(null);
                }}
                disabled={isCreating}
              >
                Hủy
              </Button>
            </div>

            {/* Progress Bar */}
            {isCreating && (
              <div className="space-y-1.5 animate-slide-up">
                <div className="relative w-full h-2 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="absolute top-0 left-0 h-full bg-gradient-to-r from-emerald-500 via-teal-500 to-emerald-500 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${Math.max(createProgress, 5)}%` }}
                  />
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-pulse" />
                </div>
                <div className="flex justify-between items-center">
                  <p className="text-xs text-muted-foreground">{createMessage}</p>
                  <span className="text-xs font-mono text-emerald-400">{createProgress}%</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="p-3 rounded-xl bg-destructive/10 border border-destructive/20 text-xs text-destructive animate-slide-up">
            {error}
          </div>
        )}

        {/* Profile List */}
        {profiles.length === 0 ? (
          <div className="text-center py-6 text-xs text-muted-foreground">
            <User className="w-8 h-8 mx-auto mb-2 opacity-30" />
            <p>Chưa có giọng mẫu nào</p>
            <p className="mt-1">Cắt audio mẫu rồi bấm &quot;Tạo mới&quot;</p>
          </div>
        ) : (
          <div className="space-y-2">
            {profiles.map((profile) => {
              const badge = getQualityBadge(profile.quality_score);
              return (
                <div
                  key={profile.id}
                  onClick={() => onSelectProfile(
                    selectedProfileId === profile.id ? null : profile.id
                  )}
                  className={`relative flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all duration-200 group ${
                    selectedProfileId === profile.id
                      ? "bg-primary/10 border-primary/30 shadow-sm shadow-primary/10"
                      : "bg-secondary/30 border-border/50 hover:bg-secondary/50 hover:border-border"
                  }`}
                >
                  {/* Avatar */}
                  <div
                    className={`flex items-center justify-center w-9 h-9 rounded-lg shrink-0 ${
                      selectedProfileId === profile.id
                        ? "bg-primary/20 text-primary"
                        : "bg-secondary text-muted-foreground"
                    }`}
                  >
                    {selectedProfileId === profile.id ? (
                      <CheckCircle2 className="w-4 h-4" />
                    ) : (
                      <User className="w-4 h-4" />
                    )}
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{profile.name}</p>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground flex-wrap">
                      <span>{formatDate(profile.created_at)}</span>
                      {/* Engine badge */}
                      {profile.engine && (
                        <span className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md text-[10px] font-medium border ${
                          profile.engine === "f5-tts"
                            ? "text-emerald-400 bg-emerald-400/10 border-emerald-400/20"
                            : "text-violet-400 bg-violet-400/10 border-violet-400/20"
                        }`}>
                          <Cpu className="w-2.5 h-2.5" />
                          {profile.engine === "f5-tts" ? "F5" : "VN"}
                        </span>
                      )}
                      {/* Quality badge */}
                      {badge && (
                        <span className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md text-[10px] font-medium border ${badge.color}`}>
                          <Shield className="w-2.5 h-2.5" />
                          {badge.label}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        togglePlay(profile);
                      }}
                      className="p-1.5 rounded-lg hover:bg-secondary/80 transition-colors text-muted-foreground hover:text-foreground"
                      title="Nghe calibration"
                    >
                      {playingId === profile.id ? (
                        <Pause className="w-3.5 h-3.5" />
                      ) : (
                        <Play className="w-3.5 h-3.5" />
                      )}
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(profile.id);
                      }}
                      className="p-1.5 rounded-lg hover:bg-destructive/10 transition-colors text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100"
                      title="Xóa profile"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        onEnded={() => setPlayingId(null)}
        onPause={() => setPlayingId(null)}
        style={{ display: "none" }}
      />
    </>
  );
}
