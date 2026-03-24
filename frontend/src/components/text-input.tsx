"use client";

import React, { useState } from "react";
import { Type, Info } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";

interface TextInputProps {
  text: string;
  setText: (text: string) => void;
  refText: string;
  setRefText: (text: string) => void;
  disabled?: boolean;
}

const MAX_CHARS = 2000;

export default function TextInput({
  text,
  setText,
  refText,
  setRefText,
  disabled,
}: TextInputProps) {
  const [showRefText, setShowRefText] = useState(false);

  return (
    <div className="space-y-4 animate-slide-up">
      {/* Main Text Input */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-sm font-medium text-foreground">
            <Type className="w-4 h-4 text-primary" />
            Văn bản cần chuyển thành giọng nói
          </label>
          <span
            className={`text-xs font-mono ${
              text.length > MAX_CHARS * 0.9
                ? "text-destructive"
                : "text-muted-foreground"
            }`}
          >
            {text.length}/{MAX_CHARS}
          </span>
        </div>
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
          placeholder="Nhập văn bản tiếng Việt tại đây... Ví dụ: Xin chào, tôi là một trợ lý ảo được tạo bởi VN-VoiceClone Pro."
          className="min-h-[140px]"
          disabled={disabled}
        />
      </div>

      {/* Reference Text (Optional) */}
      <div className="space-y-2">
        <button
          onClick={() => setShowRefText(!showRefText)}
          className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          <Info className="w-3.5 h-3.5" />
          <span>
            {showRefText
              ? "Ẩn văn bản tham chiếu"
              : "Thêm văn bản tham chiếu (không bắt buộc)"}
          </span>
        </button>

        {showRefText && (
          <div className="animate-slide-up">
            <Textarea
              value={refText}
              onChange={(e) => setRefText(e.target.value)}
              placeholder="Nhập nội dung nói trong đoạn audio mẫu (giúp cải thiện chất lượng clone)..."
              className="min-h-[80px] text-xs"
              disabled={disabled}
            />
            <p className="text-xs text-muted-foreground mt-1">
              💡 Nhập lại nội dung đang được nói trong đoạn audio mẫu sẽ giúp mô hình clone giọng chính xác hơn.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
