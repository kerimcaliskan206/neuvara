"use client";

import { useCallback, useRef, useState } from "react";

import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { config } from "@/lib/config";
import { cn, formatBytes } from "@/lib/utils";

export interface SelectedFile {
  file: File;
  previewUrl: string;
}

export interface ImageUploaderProps {
  onSelect: (selected: SelectedFile) => void;
  disabled?: boolean;
  className?: string;
}

/**
 * Drag-and-drop image picker.
 *
 * Validation runs client-side for instant feedback (MIME type + file size)
 * but the backend re-validates everything — never trust this gate alone.
 */
export function ImageUploader({ onSelect, disabled, className }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validate = useCallback((file: File): string | null => {
    if (!config.upload.allowedMimeTypes.includes(file.type)) {
      return `Desteklenmeyen dosya türü: ${file.type || "bilinmiyor"}.`;
    }
    const maxBytes = config.upload.maxMb * 1024 * 1024;
    if (file.size > maxBytes) {
      return `Dosya çok büyük (${formatBytes(file.size)}). Maks. ${config.upload.maxMb} MB.`;
    }
    if (file.size === 0) return "Dosya boş.";
    return null;
  }, []);

  const handleFile = useCallback(
    (file: File) => {
      const message = validate(file);
      if (message) {
        setError(message);
        return;
      }
      setError(null);
      const previewUrl = URL.createObjectURL(file);
      onSelect({ file, previewUrl });
    },
    [onSelect, validate],
  );

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      const file = event.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [disabled, handleFile],
  );

  return (
    <div className={cn("space-y-3", className)}>
      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        className={cn(
          "flex h-48 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed text-center transition-colors",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-border bg-muted/30 hover:bg-muted/50",
          disabled && "cursor-not-allowed opacity-60",
        )}
      >
        <p className="text-sm font-medium text-foreground">
          Görüntüyü buraya bırakın veya tıklayın
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          {config.upload.allowedMimeTypes.join(", ")} · Maks. {config.upload.maxMb} MB
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={config.upload.allowedMimeTypes.join(",")}
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) handleFile(file);
          event.target.value = ""; // allow re-selecting the same file
        }}
        disabled={disabled}
      />

      {error ? <Alert variant="danger">{error}</Alert> : null}

      <div className="flex justify-end">
        <Button
          variant="ghost"
          size="sm"
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
        >
          Dosya seç
        </Button>
      </div>
    </div>
  );
}
