import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin", "vietnamese"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "VN-VoiceClone Pro | Clone Giọng Nói Tiếng Việt",
  description:
    "Ứng dụng clone giọng nói tiếng Việt với AI. Sử dụng VieNeu-TTS để chuyển văn bản thành giọng nói tự nhiên dựa trên mẫu audio của bạn.",
  keywords: ["voice cloning", "Vietnamese TTS", "VieNeu-TTS", "tiếng Việt"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi" className="dark">
      <body className={`${inter.variable} font-sans`}>{children}</body>
    </html>
  );
}
