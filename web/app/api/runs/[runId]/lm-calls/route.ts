import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { emitRunUpdate } from "@/lib/sse-emitter";

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

    // Create LM calls in batch
    const created = await prisma.lmCall.createMany({
      data: lmCalls.map((lm: {
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
      }) => ({
        runId,
        callId: lm.callId,
        model: lm.model ?? null,
        startTime: lm.startTime,
        endTime: lm.endTime ?? null,
        durationMs: lm.durationMs ?? null,
        iteration: lm.iteration ?? null,
        phase: lm.phase ?? null,
        candidateIdx: lm.candidateIdx ?? null,
        inputs: lm.inputs ? JSON.stringify(lm.inputs) : null,
        outputs: lm.outputs ? JSON.stringify(lm.outputs) : null,
      })),
    });

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
        count: created.count,
        totalLmCalls,
      },
    });

    return NextResponse.json({ created: created.count }, { status: 201 });
  } catch (error) {
    console.error("Error pushing LM calls:", error);
    return NextResponse.json(
      { error: "Failed to push LM calls" },
      { status: 500 }
    );
  }
}
