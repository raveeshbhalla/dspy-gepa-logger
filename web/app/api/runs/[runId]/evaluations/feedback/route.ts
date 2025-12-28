import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { emitRunUpdate } from "@/lib/sse-emitter";

type RouteContext = {
  params: Promise<{ runId: string }>;
};

// PATCH /api/runs/[runId]/evaluations/feedback - Update feedback for evaluations
export async function PATCH(request: NextRequest, context: RouteContext) {
  try {
    const { runId } = await context.params;
    const body = await request.json();
    const { updates } = body;

    if (!Array.isArray(updates) || updates.length === 0) {
      return NextResponse.json(
        { error: "updates array is required" },
        { status: 400 }
      );
    }

    // Update feedback for each matching evaluation
    let updatedCount = 0;
    for (const update of updates as Array<{
      exampleId: string;
      candidateIdx: number;
      iteration: number;
      feedback: string;
    }>) {
      // Find and update the matching evaluation
      // Match by exampleId, candidateIdx, iteration, and runId
      const result = await prisma.evaluation.updateMany({
        where: {
          runId,
          exampleId: update.exampleId,
          candidateIdx: update.candidateIdx,
          iteration: update.iteration,
        },
        data: {
          feedback: update.feedback,
        },
      });
      updatedCount += result.count;
    }

    // Emit SSE event if any updates were made
    if (updatedCount > 0) {
      emitRunUpdate(runId, {
        type: "evaluation_feedback",
        data: {
          count: updatedCount,
        },
      });
    }

    return NextResponse.json({ updated: updatedCount }, { status: 200 });
  } catch (error) {
    console.error("Error updating evaluation feedback:", error);
    return NextResponse.json(
      { error: "Failed to update evaluation feedback" },
      { status: 500 }
    );
  }
}
