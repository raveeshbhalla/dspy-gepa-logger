import { PrismaClient } from "@prisma/client";
import { PrismaLibSql } from "@prisma/adapter-libsql";
import path from "path";

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

function createPrismaClient() {
  let databaseUrl = process.env.DATABASE_URL;

  if (!databaseUrl) {
    throw new Error("DATABASE_URL environment variable is not set");
  }

  // Convert relative file: URLs to absolute paths for libsql
  // DATABASE_URL "file:./dev.db" is relative to web/, not web/prisma/
  if (databaseUrl.startsWith("file:./")) {
    const relativePath = databaseUrl.replace("file:./", "");
    const absolutePath = path.resolve(process.cwd(), relativePath);
    databaseUrl = `file:${absolutePath}`;
  }

  // Create adapter with URL config (Prisma 7 API)
  const adapter = new PrismaLibSql({
    url: databaseUrl,
  });

  // Create Prisma client with adapter
  return new PrismaClient({
    adapter,
    log: process.env.NODE_ENV === "development" ? ["query", "error", "warn"] : ["error"],
  });
}

export const prisma = globalForPrisma.prisma ?? createPrismaClient();

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;

export default prisma;
