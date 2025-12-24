import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";

type RouteContext = {
  params: Promise<{ projectId: string }>;
};

// GET /api/projects/[projectId] - Get a single project with runs
export async function GET(request: NextRequest, context: RouteContext) {
  try {
    const { projectId } = await context.params;

    const project = await prisma.project.findUnique({
      where: { id: projectId },
      include: {
        runs: {
          orderBy: { startedAt: "desc" },
          select: {
            id: true,
            name: true,
            status: true,
            startedAt: true,
            completedAt: true,
            totalIterations: true,
            totalCandidates: true,
            totalLmCalls: true,
            totalEvaluations: true,
            seedScore: true,
            bestScore: true,
          },
        },
      },
    });

    if (!project) {
      return NextResponse.json(
        { error: "Project not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(project);
  } catch (error) {
    console.error("Error fetching project:", error);
    return NextResponse.json(
      { error: "Failed to fetch project" },
      { status: 500 }
    );
  }
}

// DELETE /api/projects/[projectId] - Delete a project
export async function DELETE(request: NextRequest, context: RouteContext) {
  try {
    const { projectId } = await context.params;

    await prisma.project.delete({
      where: { id: projectId },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting project:", error);
    return NextResponse.json(
      { error: "Failed to delete project" },
      { status: 500 }
    );
  }
}
