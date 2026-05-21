"use client";

import { useMutation } from "@tanstack/react-query";

import { aiAssistant } from "@/lib/api/ai";
import type {
  FusionInterpretationRequest,
  FusionResponse,
  InterpretationResponse,
  MLInterpretationRequest,
  VisionInterpretationRequest,
  VisionPredictionResponse,
} from "@/lib/api/types";

function toVisionInterpretation(
  prediction: VisionPredictionResponse,
): VisionInterpretationRequest {
  return {
    accepted: prediction.accepted,
    predicted_class: prediction.predicted_class,
    confidence: prediction.confidence,
    threshold: prediction.threshold,
    rejection_reason: prediction.rejection_reason,
    gate: {
      enabled: prediction.gate.enabled,
      predicted_class: prediction.gate.predicted_class,
      confidence: prediction.gate.confidence,
    },
    model_name: prediction.model_name,
    model_version: prediction.model_version,
  };
}

function toFusionInterpretation(
  fusion: FusionResponse,
): FusionInterpretationRequest {
  return {
    fusion_confidence: fusion.fusion_confidence,
    explanation_payload: fusion.explanation_payload,
  };
}

export function useExplainVision() {
  return useMutation<InterpretationResponse, Error, VisionPredictionResponse>({
    mutationFn: (prediction) =>
      aiAssistant.explainVision(toVisionInterpretation(prediction)),
  });
}

export function useExplainMl() {
  return useMutation<InterpretationResponse, Error, MLInterpretationRequest>({
    mutationFn: (payload) => aiAssistant.explainMl(payload),
  });
}

export function useExplainFusion() {
  return useMutation<InterpretationResponse, Error, FusionResponse>({
    mutationFn: (fusion) =>
      aiAssistant.explainFusion(toFusionInterpretation(fusion)),
  });
}
