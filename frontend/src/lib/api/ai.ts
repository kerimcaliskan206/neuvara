import { aiApi } from "@/lib/api/client";
import type {
  AIHealthResponse,
  ChatRequest,
  ChatResponse,
  FusionInterpretationRequest,
  InterpretationResponse,
  MLInterpretationRequest,
  VisionInterpretationRequest,
} from "@/lib/api/types";

export const aiAssistant = {
  async chat(payload: ChatRequest): Promise<ChatResponse> {
    const { data } = await aiApi.post<ChatResponse>("/chat", payload);
    return data;
  },

  async explainMl(payload: MLInterpretationRequest): Promise<InterpretationResponse> {
    const { data } = await aiApi.post<InterpretationResponse>("/explain/ml", payload);
    return data;
  },

  async explainVision(
    payload: VisionInterpretationRequest,
  ): Promise<InterpretationResponse> {
    const { data } = await aiApi.post<InterpretationResponse>("/explain/vision", payload);
    return data;
  },

  async explainFusion(
    payload: FusionInterpretationRequest,
  ): Promise<InterpretationResponse> {
    const { data } = await aiApi.post<InterpretationResponse>("/explain/fusion", payload);
    return data;
  },

  async health(): Promise<AIHealthResponse> {
    const { data } = await aiApi.get<AIHealthResponse>("/health");
    return data;
  },
};
