import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { emitRunUpdate } from "@/lib/sse-emitter";

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

    // Create evaluations in batch
    const created = await prisma.evaluation.createMany({
      data: evaluations.map((ev: {
        evalId: string;
        exampleId: string;
        candidateIdx?: number;
        iteration?: number;
        phase: string;
        score: number;
        feedback?: string;
        exampleInputs?: Record<string, unknown>;
        predictionPreview?: string;
        predictionRef?: string;
        timestamp: number;
      }) => ({
        runId,
        evalId: ev.evalId,
        exampleId: ev.exampleId,
        candidateIdx: ev.candidateIdx ?? null,
        iteration: ev.iteration ?? null,
        phase: ev.phase,
        score: ev.score,
        feedback: ev.feedback ?? null,
        exampleInputs: ev.exampleInputs ? JSON.stringify(ev.exampleInputs) : null,
        predictionPreview: ev.predictionPreview ?? null,
        predictionRef: ev.predictionRef ?? null,
        timestamp: ev.timestamp,
      })),
    });

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
        count: created.count,
        totalEvaluations: totalEvals,
      },
    });

    return NextResponse.json({ created: created.count }, { status: 201 });
  } catch (error) {
    console.error("Error pushing evaluations:", error);
    return NextResponse.json(
      { error: "Failed to push evaluations" },
      { status: 500 }
    );
  }
}
