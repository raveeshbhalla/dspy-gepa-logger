"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { StatsCards } from "@/components/runs/StatsCards";
import { PromptComparison } from "@/components/runs/PromptComparison";
import { PerformanceTable } from "@/components/runs/PerformanceTable";
import { Badge } from "@/components/ui/badge";

type Run = {
  id: string;
  name: string | null;
  status: "RUNNING" | "COMPLETED" | "FAILED";
  startedAt: string;
  completedAt: string | null;
  totalIterations: number;
  totalCandidates: number;
  totalLmCalls: number;
  totalEvaluations: number;
  seedPrompt: Record<string, string> | null;
  bestPrompt: Record<string, string> | null;
  bestCandidateIdx: number | null;
  seedScore: number | null;
  bestScore: number | null;
  valsetExampleIds: string[] | null;
  project: { id: string; name: string };
  candidates: Array<{
    candidateIdx: number;
    content: Record<string, string>;
    parentIdx: number | null;
  }>;
  evaluations: Array<{
    evalId: string;
    exampleId: string;
    candidateIdx: number | null;
    score: number;
    feedback: string | null;
    exampleInputs: Record<string, unknown> | null;
    predictionPreview: string | null;
    predictionRef: Record<string, unknown> | null;
    timestamp: number;
  }>;
  iterations: Array<{
    iterationNumber: number;
    totalEvals: number;
    paretoSize: number;
  }>;
};

export default function RunPage() {
  const params = useParams();
  const runId = params?.runId as string;
  const [run, setRun] = useState<Run | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;

    async function fetchRun() {
      try {
        const res = await fetch(`/api/runs/${runId}`);
        if (!res.ok) throw new Error("Run not found");
        const data = await res.json();
        setRun(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load run");
      } finally {
        setLoading(false);
      }
    }

    fetchRun();

    // Poll for updates if running
    const interval = setInterval(() => {
      if (run?.status === "RUNNING") {
        fetchRun();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [runId, run?.status]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Loading run...</p>
      </div>
    );
  }

  if (error || !run) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-destructive">{error || "Run not found"}</p>
      </div>
    );
  }

  const statusVariants: Record<string, "default" | "secondary" | "destructive"> = {
    RUNNING: "default",
    COMPLETED: "secondary",
    FAILED: "destructive",
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold text-foreground">
              {run.name || `Run ${run.id.slice(0, 8)}`}
            </h1>
            <Badge variant={statusVariants[run.status]}>
              {run.status === "RUNNING" && (
                <span className="w-2 h-2 rounded-full bg-current mr-1.5 animate-pulse" />
              )}
              {run.status}
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            {run.project.name} · Started {new Date(run.startedAt).toLocaleString()}
          </p>
        </div>
      </div>

      {/* Stats */}
      <StatsCards
        iterations={run.totalIterations}
        candidates={run.totalCandidates}
        lmCalls={run.totalLmCalls}
        evaluations={run.totalEvaluations}
        seedScore={run.seedScore}
        bestScore={run.bestScore}
      />

      {/* Prompt Comparison */}
      {run.seedPrompt && (
        <PromptComparison
          seedPrompt={run.seedPrompt}
          bestPrompt={run.bestPrompt || run.seedPrompt}
          seedCandidateIdx={0}
          bestCandidateIdx={run.bestCandidateIdx || 0}
        />
      )}

      {/* Performance Table */}
      <PerformanceTable
        evaluations={run.evaluations}
        seedCandidateIdx={0}
        bestCandidateIdx={run.bestCandidateIdx || 0}
        valsetExampleIds={run.valsetExampleIds}
      />
    </div>
  );
}
