import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { emitRunUpdate } from "@/lib/sse-emitter";
import { randomUUID } from "crypto";

type RouteContext = {
  params: Promise<{ runId: string }>;
};

type LogEntry = {
  logType: string;    // "stdout", "stderr", "lm_call", "info"
  content: string;    // Plain text or JSON string
  timestamp: number;  // Unix timestamp
  iteration?: number;
  phase?: string;
};

// POST /api/runs/[runId]/logs - Push log entries
export async function POST(request: NextRequest, context: RouteContext) {
  try {
    const { runId } = await context.params;
    const body = await request.json();
    const { logs } = body;

    if (!Array.isArray(logs) || logs.length === 0) {
      return NextResponse.json(
        { error: "logs array is required" },
        { status: 400 }
      );
    }

    // Insert all log entries
    let insertedCount = 0;
    for (const log of logs as LogEntry[]) {
      const id = randomUUID().replace(/-/g, "").slice(0, 25);
      await prisma.$executeRawUnsafe(`
        INSERT INTO Log (
          id, runId, logType, content, timestamp, iteration, phase, createdAt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
      `,
        id,
        runId,
        log.logType,
        log.content,
        log.timestamp,
        log.iteration ?? null,
        log.phase ?? null
      );
      insertedCount++;

      // Emit SSE event for each log entry (real-time streaming)
      emitRunUpdate(runId, {
        type: "log",
        data: {
          id,
          logType: log.logType,
          content: log.content,
          timestamp: log.timestamp,
          iteration: log.iteration,
          phase: log.phase,
        },
      });
    }

    return NextResponse.json({ created: insertedCount }, { status: 201 });
  } catch (error) {
    console.error("Error pushing logs:", error);
    return NextResponse.json(
      { error: "Failed to push logs" },
      { status: 500 }
    );
  }
}

// GET /api/runs/[runId]/logs - Get logs for a run
export async function GET(request: NextRequest, context: RouteContext) {
  try {
    const { runId } = await context.params;
    const { searchParams } = new URL(request.url);

    const logType = searchParams.get("logType");
    const limit = parseInt(searchParams.get("limit") || "1000");
    const afterTimestamp = parseFloat(searchParams.get("after") || "0");

    const where: { runId: string; logType?: string; timestamp?: { gt: number } } = { runId };
    if (logType) {
      where.logType = logType;
    }
    if (afterTimestamp > 0) {
      where.timestamp = { gt: afterTimestamp };
    }

    const logs = await prisma.log.findMany({
      where,
      orderBy: { timestamp: "asc" },
      take: limit,
    });

    return NextResponse.json(logs);
  } catch (error) {
    console.error("Error fetching logs:", error);
    return NextResponse.json(
      { error: "Failed to fetch logs" },
      { status: 500 }
    );
  }
}
