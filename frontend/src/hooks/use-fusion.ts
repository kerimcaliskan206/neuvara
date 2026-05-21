"use client";

import { useMutation } from "@tanstack/react-query";

import { fusionApi } from "@/lib/api/fusion";
import type { FusionRequest, FusionResponse } from "@/lib/api/types";

export function useFusionPredict() {
  return useMutation<FusionResponse, Error, FusionRequest>({
    mutationFn: (payload) => fusionApi.predict(payload),
  });
}
