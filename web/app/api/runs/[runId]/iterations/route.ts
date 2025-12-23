import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { emitRunUpdate } from "@/lib/sse-emitter";

type RouteContext = {
  params: Promise<{ runId: string }>;
};

// POST /api/runs/[runId]/iterations - Push iteration data
export async function POST(request: NextRequest, context: RouteContext) {
  try {
    const { runId } = await context.params;
    const body = await request.json();
    const {
      iterationNumber,
      timestamp,
      totalEvals,
      numCandidates,
      paretoSize,
      paretoFrontier,
      paretoPrograms,
    } = body;

    // Create or update iteration
    const iteration = await prisma.iteration.upsert({
      where: {
        runId_iterationNumber: {
          runId,
          iterationNumber,
        },
      },
      update: {
        timestamp,
        totalEvals,
        numCandidates: numCandidates ?? 0,
        paretoSize: paretoSize ?? 0,
        paretoFrontier: paretoFrontier ? JSON.stringify(paretoFrontier) : null,
        paretoPrograms: paretoPrograms ? JSON.stringify(paretoPrograms) : null,
      },
      create: {
        runId,
        iterationNumber,
        timestamp,
        totalEvals,
        numCandidates: numCandidates ?? 0,
        paretoSize: paretoSize ?? 0,
        paretoFrontier: paretoFrontier ? JSON.stringify(paretoFrontier) : null,
        paretoPrograms: paretoPrograms ? JSON.stringify(paretoPrograms) : null,
      },
    });

    // Update run stats
    await prisma.run.update({
      where: { id: runId },
      data: {
        totalIterations: iterationNumber + 1,
      },
    });

    // Emit SSE event
    emitRunUpdate(runId, {
      type: "iteration",
      data: {
        iterationNumber,
        totalEvals,
        numCandidates,
        paretoSize,
      },
    });

    return NextResponse.json(iteration, { status: 201 });
  } catch (error) {
    console.error("Error pushing iteration:", error);
    return NextResponse.json(
      { error: "Failed to push iteration" },
      { status: 500 }
    );
  }
}
