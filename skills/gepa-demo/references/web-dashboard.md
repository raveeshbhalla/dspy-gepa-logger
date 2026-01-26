# Web Dashboard Setup

From repo root:

1) Install dependencies
```bash
cd web
npm install
```

Node.js 20.19+ is required for Prisma. If you see an engine error, upgrade Node.

2) Create `web/.env`
```
DATABASE_URL="file:./dev.db"
```

3) Initialize database
```bash
npx prisma generate
npx prisma migrate deploy
```

4) Start server
```bash
npm run dev
```

Server runs at http://localhost:3000

If you see "listen EPERM" when starting the server, your shell may be sandboxed.
Retry with elevated permissions or run the command in a user shell.
