import { redirect } from "next/navigation";

export default function RootRedirect(): never {
  redirect("/login");
}
