"use client";

import { useEffect, useRef } from "react";

import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { ChatInput } from "@/components/chat/chat-input";
import { MessageBubble } from "@/components/chat/message-bubble";
import { useAiChat } from "@/hooks/use-ai-chat";

export function ChatWindow() {
  const { messages, isSending, error, hasHydrated, sendMessage, reset } = useAiChat();
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  return (
    <Card className="flex h-[70vh] min-h-[480px] flex-col">
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <div>
          <CardTitle>AI Asistan</CardTitle>
          <p className="text-xs text-muted-foreground">
            Sadece NEURAVA konularında yanıt verir. Tıbbi teşhis yerine geçmez.
          </p>
        </div>
        <Button type="button" variant="ghost" size="sm" onClick={reset}>
          Sohbeti temizle
        </Button>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-4 overflow-hidden">
        <div className="flex-1 space-y-3 overflow-y-auto pr-1">
          {!hasHydrated ? (
            <div className="space-y-3">
              <Skeleton className="h-16 w-2/3" />
              <Skeleton className="ml-auto h-12 w-1/2" />
              <Skeleton className="h-16 w-3/4" />
            </div>
          ) : null}
          {hasHydrated && messages.length === 0 ? (
            <p className="rounded-lg bg-muted/50 p-4 text-sm text-muted-foreground">
              Bir soru yazın. Örnek: &ldquo;Modelin pozitif tahmin etmesi ne anlama
              geliyor?&rdquo;
            </p>
          ) : null}
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          {isSending ? (
            <div className="flex">
              <div className="rounded-2xl border border-border bg-background px-4 py-3">
                <Spinner size="sm" label="Yanıt yazılıyor..." />
              </div>
            </div>
          ) : null}
          <div ref={endRef} />
        </div>

        {error ? <Alert variant="danger">{error}</Alert> : null}

        <ChatInput onSend={sendMessage} disabled={isSending} />
      </CardContent>
    </Card>
  );
}
