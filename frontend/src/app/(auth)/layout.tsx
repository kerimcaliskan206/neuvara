import { AuthScene } from "@/components/auth/auth-scene";

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return <AuthScene>{children}</AuthScene>;
}
