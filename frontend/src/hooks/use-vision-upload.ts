"use client";

import { useCallback, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { visionApi, type VisionPredictOptions } from "@/lib/api/vision";
import type { VisionPredictionResponse } from "@/lib/api/types";

export interface VisionUploadVariables {
  file: File;
  gradcam?: boolean;
  threshold?: number;
}

/**
 * Vision upload mutation with progress + cancellation.
 *
 * `progress` is 0..100 during upload, then null after the response.  The
 * AbortController is recreated per mutate call so consecutive uploads
 * don't share a cancel signal.
 */
export function useVisionUpload() {
  const [progress, setProgress] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const mutation = useMutation<VisionPredictionResponse, Error, VisionUploadVariables>({
    mutationFn: async ({ file, ...options }) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setProgress(0);

      const apiOptions: VisionPredictOptions = {
        gradcam: options.gradcam,
        threshold: options.threshold,
        signal: controller.signal,
        onProgress: (percent) => setProgress(percent),
      };
      try {
        return await visionApi.predict(file, apiOptions);
      } finally {
        setProgress(null);
      }
    },
  });

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setProgress(null);
  }, []);

  return Object.assign(mutation, { progress, cancel });
}
