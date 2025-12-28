import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";

type RouteContext = {
  params: Promise<{ runId: string }>;
};

// GET /api/runs/[runId] - Get full run details
export async function GET(request: NextRequest, context: RouteContext) {
  try {
    const { runId } = await context.params;

    const run = await prisma.run.findUnique({
      where: { id: runId },
      include: {
        project: {
          select: { id: true, name: true },
        },
        iterations: {
          orderBy: { iterationNumber: "asc" },
        },
        candidates: {
          orderBy: { candidateIdx: "asc" },
        },
        evaluations: {
          orderBy: { timestamp: "asc" },
        },
        lmCalls: {
          orderBy: { startTime: "asc" },
        },
      },
    });

    if (!run) {
      return NextResponse.json({ error: "Run not found" }, { status: 404 });
    }

    // Parse JSON fields
    const parsedRun = {
      ...run,
      config: run.config ? JSON.parse(run.config) : null,
      seedPrompt: run.seedPrompt ? JSON.parse(run.seedPrompt) : null,
      bestPrompt: run.bestPrompt ? JSON.parse(run.bestPrompt) : null,
      valsetExampleIds: run.valsetExampleIds ? JSON.parse(run.valsetExampleIds) : null,
      iterations: run.iterations.map((iter: typeof run.iterations[number]) => ({
        ...iter,
        paretoFrontier: iter.paretoFrontier ? JSON.parse(iter.paretoFrontier) : null,
        paretoPrograms: iter.paretoPrograms ? JSON.parse(iter.paretoPrograms) : null,
      })),
      candidates: run.candidates.map((cand: typeof run.candidates[number]) => ({
        ...cand,
        content: JSON.parse(cand.content),
      })),
      evaluations: run.evaluations.map((ev: typeof run.evaluations[number]) => {
        let parsedPredictionRef = null;
        if (ev.predictionRef) {
          try {
            parsedPredictionRef = JSON.parse(ev.predictionRef);
          } catch {
            // If predictionRef is not valid JSON, return as raw string
            parsedPredictionRef = ev.predictionRef;
          }
        }
        return {
          ...ev,
          exampleInputs: ev.exampleInputs ? JSON.parse(ev.exampleInputs) : null,
          predictionRef: parsedPredictionRef,
        };
      }),
      lmCalls: run.lmCalls.map((lm: typeof run.lmCalls[number]) => ({
        ...lm,
        inputs: lm.inputs ? JSON.parse(lm.inputs) : null,
        outputs: lm.outputs ? JSON.parse(lm.outputs) : null,
      })),
    };

    return NextResponse.json(parsedRun);
  } catch (error) {
    console.error("Error fetching run:", error);
    return NextResponse.json(
      { error: "Failed to fetch run" },
      { status: 500 }
    );
  }
}

// PATCH /api/runs/[runId] - Update run metadata (seed_prompt, valset, etc.)
export async function PATCH(request: NextRequest, context: RouteContext) {
  try {
    const { runId } = await context.params;
    const body = await request.json();
    const { seedPrompt, valsetExampleIds } = body;

    const updateData: Record<string, unknown> = {};

    if (seedPrompt !== undefined) {
      updateData.seedPrompt = seedPrompt ? JSON.stringify(seedPrompt) : null;
    }

    if (valsetExampleIds !== undefined) {
      updateData.valsetExampleIds = valsetExampleIds
        ? JSON.stringify(valsetExampleIds)
        : null;
    }

    const run = await prisma.run.update({
      where: { id: runId },
      data: updateData,
    });

    // If we have a seed prompt and no candidate 0 yet, create it
    if (seedPrompt) {
      const existingCandidate = await prisma.candidate.findUnique({
        where: {
          runId_candidateIdx: {
            runId,
            candidateIdx: 0,
          },
        },
      });

      if (!existingCandidate) {
        await prisma.candidate.create({
          data: {
            runId: run.id,
            candidateIdx: 0,
            content: JSON.stringify(seedPrompt),
            parentIdx: null,
            createdAtIter: 0,
          },
        });

        await prisma.run.update({
          where: { id: run.id },
          data: { totalCandidates: { increment: 1 } },
        });
      }
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error updating run:", error);
    return NextResponse.json(
      { error: "Failed to update run" },
      { status: 500 }
    );
  }
}

// DELETE /api/runs/[runId] - Delete a run
export async function DELETE(request: NextRequest, context: RouteContext) {
  try {
    const { runId } = await context.params;

    await prisma.run.delete({
      where: { id: runId },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting run:", error);
    return NextResponse.json(
      { error: "Failed to delete run" },
      { status: 500 }
    );
  }
}
