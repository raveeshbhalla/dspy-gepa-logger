import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { emitRunUpdate, emitGlobalEvent } from "@/lib/sse-emitter";

type RouteContext = {
  params: Promise<{ runId: string }>;
};

// PUT /api/runs/[runId]/status - Update run status (complete/fail)
export async function PUT(request: NextRequest, context: RouteContext) {
  try {
    const { runId } = await context.params;
    const body = await request.json();
    const {
      status,
      bestPrompt,
      bestCandidateIdx,
      bestScore,
      seedScore,
    } = body;

    if (!status || !["COMPLETED", "FAILED"].includes(status)) {
      return NextResponse.json(
        { error: "status must be COMPLETED or FAILED" },
        { status: 400 }
      );
    }

    const run = await prisma.run.update({
      where: { id: runId },
      data: {
        status,
        completedAt: new Date(),
        bestPrompt: bestPrompt ? JSON.stringify(bestPrompt) : undefined,
        bestCandidateIdx: bestCandidateIdx ?? undefined,
        bestScore: bestScore ?? undefined,
        seedScore: seedScore ?? undefined,
      },
    });

    // Emit SSE event for run detail page
    emitRunUpdate(runId, {
      type: "status",
      data: {
        status,
        completedAt: run.completedAt?.toISOString(),
        bestCandidateIdx,
        bestScore,
        seedScore,
      },
    });

    // Emit global event for sidebar to update
    emitGlobalEvent({
      type: "run_completed",
      data: {
        runId,
        status,
        bestScore,
      },
    });

    return NextResponse.json(run);
  } catch (error) {
    console.error("Error updating run status:", error);
    return NextResponse.json(
      { error: "Failed to update run status" },
      { status: 500 }
    );
  }
}
