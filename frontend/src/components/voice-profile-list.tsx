"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  User,
  Trash2,
  Play,
  Pause,
  Plus,
  Dna,
  Loader2,
  CheckCircle2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  VoiceProfile,
  createVoiceProfile,
  listVoiceProfiles,
  deleteVoiceProfile,
  getProfileAudioUrl,
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
  const audioRef = useRef<HTMLAudioElement>(null);

  // Load profiles on mount
  useEffect(() => {
    loadProfiles();
  }, []);

  const loadProfiles = async () => {
    try {
      const data = await listVoiceProfiles();
      setProfiles(data);
    } catch {
      console.error("Failed to load profiles");
    }
  };

  const handleCreate = async () => {
    if (!trimmedFilename || !profileName.trim()) return;

    setIsCreating(true);
    setError(null);

    try {
      const newProfile = await createVoiceProfile(
        trimmedFilename,
        profileName.trim(),
        refText || undefined,
        profileEngine
      );
      setProfiles((prev) => [newProfile, ...prev]);
      setProfileName("");
      setShowCreateForm(false);
      onSelectProfile(newProfile.id);
      onProfileCreated?.();
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "Lỗi tạo voice profile"
      );
    } finally {
      setIsCreating(false);
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
              <p className="text-xs text-emerald-400">✨ Chất lượng cao, đọc đầy đủ (chậm hơn ~30-60s)</p>
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
                    Đang tạo giọng mẫu...
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
              >
                Hủy
              </Button>
            </div>
            {isCreating && (
              <p className="text-xs text-muted-foreground animate-pulse">
                ⏳ {profileEngine === "f5-tts" ? "F5-TTS đang clone giọng nói... (~30-60s)" : "AI đang phân tích giọng nói... (~10-30s)"}
              </p>
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
            {profiles.map((profile) => (
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
                  <p className="text-xs text-muted-foreground">
                    {formatDate(profile.created_at)} • {profile.codes_count}{" "}
                    codes
                  </p>
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
            ))}
          </div>
        )}
      </div>

      {/* Hidden audio element - outside main div to avoid hydration issues */}
      <audio
        ref={audioRef}
        onEnded={() => setPlayingId(null)}
        onPause={() => setPlayingId(null)}
        style={{ display: "none" }}
      />
    </>
  );
}
