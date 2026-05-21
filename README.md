# HantaProject

Production-grade backend API built with FastAPI, PostgreSQL, SQLAlchemy, and Docker.

## Tech Stack

- **Python 3.12**
- **FastAPI** — web framework
- **PostgreSQL 16** — database
- **SQLAlchemy 2** — ORM (async)
- **Alembic** — database migrations
- **Docker / docker-compose** — containerization

---

## Project Structure

```
hantaproject/
├── app/
│   ├── main.py              # FastAPI app factory
│   ├── core/
│   │   ├── config.py        # Settings (env vars via pydantic-settings)
│   │   └── database.py      # Async SQLAlchemy engine, session, Base
│   ├── api/
│   │   └── v1/
│   │       ├── router.py    # Aggregates all v1 route groups
│   │       └── routes/
│   │           └── health.py
│   ├── models/              # SQLAlchemy ORM models
│   ├── schemas/             # Pydantic request/response schemas
│   ├── services/            # Business logic layer
│   ├── repositories/        # Data access layer
│   └── modules/             # Future feature modules
│       ├── auth/
│       ├── ml/
│       ├── ai/
│       └── vision/
├── alembic/                 # Database migrations
├── alembic.ini
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Getting Started

### 1. Clone and configure environment

```bash
git clone <repo-url>
cd hantaproject
cp .env.example .env
```

Edit `.env` with your values if needed. Defaults work out of the box with Docker.

### 2. Start with Docker (recommended)

```bash
docker-compose up --build
```

This starts:
- `hanta_api` — FastAPI app on port **8000**
- `hanta_db` — PostgreSQL on port **5432**

### 3. Run migrations

In a separate terminal (while containers are running):

```bash
docker-compose exec api alembic upgrade head
```

### 4. Verify

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/health/db
```

---

## Local Development (without Docker)

### Requirements

- Python 3.12+
- PostgreSQL running locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Update `.env` → set `POSTGRES_HOST=localhost`.

```bash
uvicorn app.main:app --reload
```

---

## Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "describe your change"

# Apply all pending migrations
alembic upgrade head

# Roll back one migration
alembic downgrade -1

# View migration history
alembic history
```

---

## API Docs

Available only when `DEBUG=true`:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `HantaProject` | Application name |
| `APP_VERSION` | `0.1.0` | Application version |
| `DEBUG` | `false` | Enable debug mode & docs |
| `ENVIRONMENT` | `development` | Runtime environment |
| `API_V1_PREFIX` | `/api/v1` | API route prefix |
| `POSTGRES_HOST` | — | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | — | PostgreSQL user |
| `POSTGRES_PASSWORD` | — | PostgreSQL password |
| `POSTGRES_DB` | — | PostgreSQL database name |
