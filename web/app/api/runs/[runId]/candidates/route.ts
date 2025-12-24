import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { emitRunUpdate } from "@/lib/sse-emitter";

type RouteContext = {
  params: Promise<{ runId: string }>;
};

// POST /api/runs/[runId]/candidates - Push candidate batch
export async function POST(request: NextRequest, context: RouteContext) {
  try {
    const { runId } = await context.params;
    const body = await request.json();
    const { candidates } = body;

    if (!Array.isArray(candidates) || candidates.length === 0) {
      return NextResponse.json(
        { error: "candidates array is required" },
        { status: 400 }
      );
    }

    // Upsert candidates (in case of duplicates)
    let created = 0;
    for (const cand of candidates) {
      await prisma.candidate.upsert({
        where: {
          runId_candidateIdx: {
            runId,
            candidateIdx: cand.candidateIdx,
          },
        },
        update: {
          content: JSON.stringify(cand.content),
          parentIdx: cand.parentIdx ?? null,
          createdAtIter: cand.createdAtIter ?? null,
        },
        create: {
          runId,
          candidateIdx: cand.candidateIdx,
          content: JSON.stringify(cand.content),
          parentIdx: cand.parentIdx ?? null,
          createdAtIter: cand.createdAtIter ?? null,
        },
      });
      created++;
    }

    // Update run stats
    const totalCandidates = await prisma.candidate.count({
      where: { runId },
    });

    await prisma.run.update({
      where: { id: runId },
      data: { totalCandidates },
    });

    // Emit SSE event
    emitRunUpdate(runId, {
      type: "candidate",
      data: {
        count: created,
        totalCandidates,
      },
    });

    return NextResponse.json({ created }, { status: 201 });
  } catch (error) {
    console.error("Error pushing candidates:", error);
    return NextResponse.json(
      { error: "Failed to push candidates" },
      { status: 500 }
    );
  }
}
