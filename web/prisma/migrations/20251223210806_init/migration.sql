-- CreateTable
CREATE TABLE "Project" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "Run" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "projectId" TEXT NOT NULL,
    "name" TEXT,
    "status" TEXT NOT NULL DEFAULT 'RUNNING',
    "startedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" DATETIME,
    "config" TEXT,
    "totalIterations" INTEGER NOT NULL DEFAULT 0,
    "totalCandidates" INTEGER NOT NULL DEFAULT 0,
    "totalLmCalls" INTEGER NOT NULL DEFAULT 0,
    "totalEvaluations" INTEGER NOT NULL DEFAULT 0,
    "seedPrompt" TEXT,
    "bestPrompt" TEXT,
    "bestCandidateIdx" INTEGER,
    "seedScore" REAL,
    "bestScore" REAL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "Run_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "Project" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "Iteration" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "runId" TEXT NOT NULL,
    "iterationNumber" INTEGER NOT NULL,
    "timestamp" REAL NOT NULL,
    "totalEvals" INTEGER NOT NULL,
    "numCandidates" INTEGER NOT NULL DEFAULT 0,
    "paretoSize" INTEGER NOT NULL DEFAULT 0,
    "paretoFrontier" TEXT,
    "paretoPrograms" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "Iteration_runId_fkey" FOREIGN KEY ("runId") REFERENCES "Run" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "Candidate" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "runId" TEXT NOT NULL,
    "candidateIdx" INTEGER NOT NULL,
    "content" TEXT NOT NULL,
    "parentIdx" INTEGER,
    "createdAtIter" INTEGER,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "Candidate_runId_fkey" FOREIGN KEY ("runId") REFERENCES "Run" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "Evaluation" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "runId" TEXT NOT NULL,
    "evalId" TEXT NOT NULL,
    "exampleId" TEXT NOT NULL,
    "candidateIdx" INTEGER,
    "iteration" INTEGER,
    "phase" TEXT NOT NULL,
    "score" REAL NOT NULL,
    "feedback" TEXT,
    "exampleInputs" TEXT,
    "predictionPreview" TEXT,
    "timestamp" REAL NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "Evaluation_runId_fkey" FOREIGN KEY ("runId") REFERENCES "Run" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "LmCall" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "runId" TEXT NOT NULL,
    "callId" TEXT NOT NULL,
    "model" TEXT,
    "startTime" REAL NOT NULL,
    "endTime" REAL,
    "durationMs" REAL,
    "iteration" INTEGER,
    "phase" TEXT,
    "candidateIdx" INTEGER,
    "inputs" TEXT,
    "outputs" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LmCall_runId_fkey" FOREIGN KEY ("runId") REFERENCES "Run" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateIndex
CREATE UNIQUE INDEX "Project_name_key" ON "Project"("name");

-- CreateIndex
CREATE INDEX "Run_projectId_idx" ON "Run"("projectId");

-- CreateIndex
CREATE INDEX "Run_status_idx" ON "Run"("status");

-- CreateIndex
CREATE INDEX "Iteration_runId_idx" ON "Iteration"("runId");

-- CreateIndex
CREATE UNIQUE INDEX "Iteration_runId_iterationNumber_key" ON "Iteration"("runId", "iterationNumber");

-- CreateIndex
CREATE INDEX "Candidate_runId_idx" ON "Candidate"("runId");

-- CreateIndex
CREATE UNIQUE INDEX "Candidate_runId_candidateIdx_key" ON "Candidate"("runId", "candidateIdx");

-- CreateIndex
CREATE INDEX "Evaluation_runId_idx" ON "Evaluation"("runId");

-- CreateIndex
CREATE INDEX "Evaluation_runId_exampleId_idx" ON "Evaluation"("runId", "exampleId");

-- CreateIndex
CREATE INDEX "Evaluation_runId_candidateIdx_idx" ON "Evaluation"("runId", "candidateIdx");

-- CreateIndex
CREATE INDEX "LmCall_runId_idx" ON "LmCall"("runId");

-- CreateIndex
CREATE INDEX "LmCall_runId_iteration_idx" ON "LmCall"("runId", "iteration");

-- CreateIndex
CREATE INDEX "LmCall_runId_phase_idx" ON "LmCall"("runId", "phase");
