import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { emitGlobalEvent } from "@/lib/sse-emitter";

// POST /api/runs - Create a new run
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { projectName = "Default", name, config, seedPrompt, valsetExampleIds } = body;

    // Ensure project exists (create if not)
    const project = await prisma.project.upsert({
      where: { name: projectName },
      update: {},
      create: { name: projectName },
    });

    // Create the run
    const run = await prisma.run.create({
      data: {
        projectId: project.id,
        name,
        config: config ? JSON.stringify(config) : null,
        seedPrompt: seedPrompt ? JSON.stringify(seedPrompt) : null,
        valsetExampleIds: valsetExampleIds ? JSON.stringify(valsetExampleIds) : null,
        status: "RUNNING",
      },
    });

    // If we have a seed prompt, create candidate 0
    if (seedPrompt) {
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
        data: { totalCandidates: 1 },
      });
    }

    // Emit global event for sidebar to update
    emitGlobalEvent({
      type: "run_created",
      data: {
        runId: run.id,
        projectId: project.id,
        name,
        status: "RUNNING",
        startedAt: run.startedAt.toISOString(),
      },
    });

    return NextResponse.json({ runId: run.id, projectId: project.id }, { status: 201 });
  } catch (error) {
    console.error("Error creating run:", error);
    return NextResponse.json(
      { error: "Failed to create run" },
      { status: 500 }
    );
  }
}

// GET /api/runs - List recent runs across all projects
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get("limit") || "20");
    const projectId = searchParams.get("projectId");

    const runs = await prisma.run.findMany({
      where: projectId ? { projectId } : undefined,
      orderBy: { startedAt: "desc" },
      take: limit,
      include: {
        project: {
          select: { name: true },
        },
      },
    });

    return NextResponse.json(runs);
  } catch (error) {
    console.error("Error fetching runs:", error);
    return NextResponse.json(
      { error: "Failed to fetch runs" },
      { status: 500 }
    );
  }
}
