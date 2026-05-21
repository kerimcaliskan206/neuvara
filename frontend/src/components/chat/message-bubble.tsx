"use client";

import { Badge } from "@/components/ui/badge";
import { cn, formatTimestamp } from "@/lib/utils";

export type ChatRole = "user" | "assistant";

export interface ChatBubbleMessage {
  id: string;
  role: ChatRole;
  content: string;
  refused?: boolean;
  refusalReason?: string | null;
  timestamp?: string;
}

export interface MessageBubbleProps {
  message: ChatBubbleMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-primary text-primary-foreground"
            : message.refused
              ? "border border-warning/40 bg-warning/10 text-foreground"
              : "border border-border bg-background text-foreground",
        )}
      >
        {message.refused ? (
          <div className="mb-1.5 flex items-center gap-1.5">
            <Badge variant="warning">Reddedildi</Badge>
            {message.refusalReason ? (
              <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
                {message.refusalReason}
              </span>
            ) : null}
          </div>
        ) : null}
        <p className="whitespace-pre-wrap">{message.content}</p>
        {message.timestamp ? (
          <p
            className={cn(
              "mt-2 text-[10px]",
              isUser ? "text-primary-foreground/70" : "text-muted-foreground",
            )}
          >
            {formatTimestamp(message.timestamp)}
          </p>
        ) : null}
      </div>
    </div>
  );
}
