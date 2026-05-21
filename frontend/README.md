# HantaProject — Frontend

Next.js (App Router) + TypeScript + Tailwind + Axios + TanStack Query + Zustand.

## Run

```bash
cp .env.local.example .env.local   # already done; tweak as needed
npm install
npm run dev                        # http://localhost:3000
```

The backend should be reachable at `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`).

## Scripts

- `npm run dev` — Next dev server
- `npm run build` — production build (used in CI)
- `npm run lint` — Next lint
- `npm run typecheck` — `tsc --noEmit`
- `npm test` — Vitest one-shot
- `npm run test:watch` — Vitest watch

## Architecture

```
src/
├── app/                # App Router pages (route groups for auth + dashboard)
├── components/
│   ├── ui/             # Reusable primitives (Button, Card, Badge, Alert, ...)
│   ├── layout/         # Sidebar, TopNav, DashboardShell
│   ├── auth/           # LoginForm, RegisterForm, ProtectedRoute
│   ├── vision/         # ImageUploader, PredictionResult, GradCamOverlay
│   └── chat/           # ChatWindow, MessageBubble, ChatInput
├── hooks/              # useAuth, useVisionUpload, useAiChat
├── lib/
│   ├── api/            # client, auth, ml, vision, ai, types
│   ├── config.ts       # typed reader for NEXT_PUBLIC_*
│   ├── logger.ts       # dev/prod-aware console wrapper
│   └── utils.ts        # cn(), formatters
├── stores/             # Zustand auth-store (with token persistence)
└── tests/              # Vitest setup + API/store specs
```

### Rules

- UI components never call axios directly — they use hooks, which use `lib/api/*`, which use the shared axios instance from `lib/api/client.ts`.
- Every URL/limit/flag comes from `lib/config.ts`. No `process.env.*` reads outside that file.
- The auth store registers a token accessor with the axios layer at import time, so every request automatically carries `Authorization: Bearer ...`. A 401 fires `auth:unauthorized`, the QueryClient provider catches it and redirects to `/login`.
- `ProtectedRoute` is a UX guard, not a security boundary — the backend is the authority.
