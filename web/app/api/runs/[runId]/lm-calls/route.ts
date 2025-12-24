import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { emitRunUpdate } from "@/lib/sse-emitter";
import { randomUUID } from "crypto";

type RouteContext = {
  params: Promise<{ runId: string }>;
};

// POST /api/runs/[runId]/lm-calls - Push LM call batch
export async function POST(request: NextRequest, context: RouteContext) {
  try {
    const { runId } = await context.params;
    const body = await request.json();
    const { lmCalls } = body;

    if (!Array.isArray(lmCalls) || lmCalls.length === 0) {
      return NextResponse.json(
        { error: "lmCalls array is required" },
        { status: 400 }
      );
    }

    // Use INSERT OR IGNORE for SQLite idempotency (skipDuplicates not supported)
    // This safely handles retries without duplicating data
    let insertedCount = 0;
    for (const lm of lmCalls as Array<{
      callId: string;
      model?: string;
      startTime: number;
      endTime?: number;
      durationMs?: number;
      iteration?: number;
      phase?: string;
      candidateIdx?: number;
      inputs?: Record<string, unknown>;
      outputs?: Record<string, unknown>;
    }>) {
      const id = randomUUID().replace(/-/g, "").slice(0, 25);
      const inputs = lm.inputs ? JSON.stringify(lm.inputs) : null;
      const outputs = lm.outputs ? JSON.stringify(lm.outputs) : null;
      const result = await prisma.$executeRawUnsafe(`
        INSERT OR IGNORE INTO LmCall (
          id, runId, callId, model, startTime, endTime, durationMs,
          iteration, phase, candidateIdx, inputs, outputs, createdAt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
      `,
        id,
        runId,
        lm.callId,
        lm.model ?? null,
        lm.startTime,
        lm.endTime ?? null,
        lm.durationMs ?? null,
        lm.iteration ?? null,
        lm.phase ?? null,
        lm.candidateIdx ?? null,
        inputs,
        outputs
      );
      insertedCount += result;
    }

    // Update run stats
    const totalLmCalls = await prisma.lmCall.count({
      where: { runId },
    });

    await prisma.run.update({
      where: { id: runId },
      data: { totalLmCalls },
    });

    // Emit SSE event
    emitRunUpdate(runId, {
      type: "lm_call",
      data: {
        count: insertedCount,
        totalLmCalls,
      },
    });

    return NextResponse.json({ created: insertedCount }, { status: 201 });
  } catch (error) {
    console.error("Error pushing LM calls:", error);
    return NextResponse.json(
      { error: "Failed to push LM calls" },
      { status: 500 }
    );
  }
}
