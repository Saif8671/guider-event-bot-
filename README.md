# Event Platform

Monorepo scaffold for a modern event platform with:
- Next.js frontend
- FastAPI backend
- PostgreSQL database
- Redis for background jobs and caching

## Layout

- `frontend/` - Next.js app router application
- `backend/` - FastAPI service
- `docker-compose.yml` - local database and Redis
- `SYSTEM_DESIGN.md` - product and architecture spec

## Local development

1. Copy `.env.example` to `.env`
2. Start infrastructure:
   ```bash
   docker compose up -d
   ```
3. Run the backend:
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload
   ```
4. Run the frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

