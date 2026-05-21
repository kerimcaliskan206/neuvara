"use client";

import { useCallback, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { medicalApi, type MedicalAnalyzeOptions } from "@/lib/api/medical";
import type { ClinicalContextRequest, UnifiedAnalysisSession } from "@/lib/api/types";

export interface MedicalAnalyzeVariables {
  file: File;
  gradcam?: boolean;
  clinicalContext?: ClinicalContextRequest | null;
}

export function useMedicalAnalysis() {
  const [progress, setProgress] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  // Tracks pending state synchronously so the locked mutate wrapper
  // can read it without waiting for a React re-render cycle.
  const isPendingRef = useRef(false);

  const mutation = useMutation<UnifiedAnalysisSession, Error, MedicalAnalyzeVariables>({
    mutationFn: async ({ file, gradcam = true, clinicalContext }) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setProgress(0);

      const opts: MedicalAnalyzeOptions = {
        gradcam,
        clinicalContext: clinicalContext ?? null,
        signal: controller.signal,
        onProgress: (pct) => setProgress(pct),
      };

      try {
        const result = await medicalApi.analyze(file, opts);
        return result;
      } finally {
        isPendingRef.current = false;
        setProgress(null);
      }
    },
  });

  // Keep ref in sync on every render so the wrapper always has the latest value.
  isPendingRef.current = mutation.isPending;

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    isPendingRef.current = false;
    setProgress(null);
  }, []);

  // Locked wrapper — drops the call synchronously if a request is already in
  // flight, preventing duplicate backend hits caused by rapid double-clicks or
  // calls fired before React re-renders with the new isPending state.
  const mutate = useCallback(
    (vars: MedicalAnalyzeVariables) => {
      if (isPendingRef.current) return;
      isPendingRef.current = true;
      mutation.mutate(vars);
    },
    // mutation.mutate is stable across renders (TanStack Query guarantee).
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [mutation.mutate],
  );

  return { ...mutation, progress, cancel, mutate };
}
