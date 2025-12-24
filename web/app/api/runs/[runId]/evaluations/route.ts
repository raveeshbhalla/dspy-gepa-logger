import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { emitRunUpdate } from "@/lib/sse-emitter";
import { randomUUID } from "crypto";

type RouteContext = {
  params: Promise<{ runId: string }>;
};

// POST /api/runs/[runId]/evaluations - Push evaluation batch
export async function POST(request: NextRequest, context: RouteContext) {
  try {
    const { runId } = await context.params;
    const body = await request.json();
    const { evaluations } = body;

    if (!Array.isArray(evaluations) || evaluations.length === 0) {
      return NextResponse.json(
        { error: "evaluations array is required" },
        { status: 400 }
      );
    }

    // Use INSERT OR IGNORE for SQLite idempotency (skipDuplicates not supported)
    // This safely handles retries without duplicating data
    let insertedCount = 0;
    for (const ev of evaluations as Array<{
      evalId: string;
      exampleId: string;
      candidateIdx?: number;
      iteration?: number;
      phase: string;
      score: number;
      feedback?: string;
      exampleInputs?: Record<string, unknown>;
      predictionPreview?: string;
      predictionRef?: unknown;
      timestamp: number;
    }>) {
      const id = randomUUID().replace(/-/g, "").slice(0, 25);
      const exampleInputs = ev.exampleInputs ? JSON.stringify(ev.exampleInputs) : null;
      // Normalize predictionRef to JSON string - handles objects sent from Python client
      const predictionRef = ev.predictionRef != null
        ? (typeof ev.predictionRef === "string"
            ? ev.predictionRef
            : JSON.stringify(ev.predictionRef))
        : null;
      const result = await prisma.$executeRawUnsafe(`
        INSERT OR IGNORE INTO Evaluation (
          id, runId, evalId, exampleId, candidateIdx, iteration, phase,
          score, feedback, exampleInputs, predictionPreview, predictionRef,
          timestamp, createdAt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
      `,
        id,
        runId,
        ev.evalId,
        ev.exampleId,
        ev.candidateIdx ?? null,
        ev.iteration ?? null,
        ev.phase,
        ev.score,
        ev.feedback ?? null,
        exampleInputs,
        ev.predictionPreview ?? null,
        predictionRef,
        ev.timestamp
      );
      insertedCount += result;
    }

    // Update run stats
    const totalEvals = await prisma.evaluation.count({
      where: { runId },
    });

    await prisma.run.update({
      where: { id: runId },
      data: { totalEvaluations: totalEvals },
    });

    // Emit SSE event
    emitRunUpdate(runId, {
      type: "evaluation",
      data: {
        count: insertedCount,
        totalEvaluations: totalEvals,
      },
    });

    return NextResponse.json({ created: insertedCount }, { status: 201 });
  } catch (error) {
    console.error("Error pushing evaluations:", error);
    return NextResponse.json(
      { error: "Failed to push evaluations" },
      { status: 500 }
    );
  }
}
