-- AlterTable: Add new columns to Iteration
ALTER TABLE "Iteration" ADD COLUMN "reflectionInput" TEXT;
ALTER TABLE "Iteration" ADD COLUMN "reflectionOutput" TEXT;
ALTER TABLE "Iteration" ADD COLUMN "proposedChanges" TEXT;
ALTER TABLE "Iteration" ADD COLUMN "parentCandidateIdx" INTEGER;
ALTER TABLE "Iteration" ADD COLUMN "childCandidateIdxs" TEXT;

-- CreateTable: Log
CREATE TABLE "Log" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "runId" TEXT NOT NULL,
    "logType" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "timestamp" REAL NOT NULL,
    "iteration" INTEGER,
    "phase" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "Log_runId_fkey" FOREIGN KEY ("runId") REFERENCES "Run" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateIndex
CREATE INDEX "Log_runId_idx" ON "Log"("runId");

-- CreateIndex
CREATE INDEX "Log_runId_logType_idx" ON "Log"("runId", "logType");
