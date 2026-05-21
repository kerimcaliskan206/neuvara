import { api } from "@/lib/api/client";
import type {
  LoginRequest,
  RegisterRequest,
  TokenResponse,
  UserResponse,
} from "@/lib/api/types";

export const authApi = {
  async register(payload: RegisterRequest): Promise<UserResponse> {
    const { data } = await api.post<UserResponse>("/auth/register", payload);
    return data;
  },

  async login(payload: LoginRequest): Promise<TokenResponse> {
    const { data } = await api.post<TokenResponse>("/auth/login", payload);
    return data;
  },

  async me(): Promise<UserResponse> {
    const { data } = await api.get<UserResponse>("/auth/me");
    return data;
  },

  async forgotPassword(email: string): Promise<void> {
    await api.post("/auth/forgot-password", { email });
  },

  async resetPassword(token: string, new_password: string): Promise<void> {
    await api.post("/auth/reset-password", { token, new_password });
  },
};
