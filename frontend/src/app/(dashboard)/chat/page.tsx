"use client";

import { Alert } from "@/components/ui/alert";
import { ChatWindow } from "@/components/chat/chat-window";
import { config } from "@/lib/config";

export default function ChatPage() {
  if (!config.features.aiChat) {
    return (
      <Alert variant="warning" title="Devre dışı">
        AI sohbet özelliği bu ortamda kapatılmıştır.
      </Alert>
    );
  }
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight text-white">AI Asistan</h1>
        <p className="mt-1 text-sm text-white/65">
          Türkçe, konu kilitli yardımcı. Yalnızca NEURAVA kapsamındaki
          sorulara yanıt verir.
        </p>
      </header>
      <ChatWindow />
    </div>
  );
}
