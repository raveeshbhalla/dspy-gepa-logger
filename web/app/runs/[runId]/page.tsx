"use client";

import { useEffect, useState, useMemo } from "react";
import { useParams } from "next/navigation";
import { StatsCards } from "@/components/runs/StatsCards";
import { PromptComparison } from "@/components/runs/PromptComparison";
import { PerformanceTable } from "@/components/runs/PerformanceTable";
import { LogsTab } from "@/components/runs/LogsTab";
import { IterationsTab } from "@/components/runs/IterationsTab";
import { LineageTab } from "@/components/runs/LineageTab";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

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
    createdAtIter?: number | null;
  }>;
  evaluations: Array<{
    evalId: string;
    exampleId: string;
    candidateIdx: number | null;
    iteration: number | null;
    phase: string;
    score: number;
    feedback: string | null;
    exampleInputs: Record<string, unknown> | null;
    predictionPreview: string | null;
    predictionRef: Record<string, unknown> | null;
    timestamp: number;
  }>;
  iterations: Array<{
    iterationNumber: number;
    timestamp: number;
    totalEvals: number;
    numCandidates: number;
    paretoSize: number;
    paretoFrontier: Record<string, number> | null;
    paretoPrograms: Record<string, number> | null;
    reflectionInput?: string | null;
    reflectionOutput?: string | null;
    proposedChanges?: string | null;
    parentCandidateIdx?: number | null;
    childCandidateIdxs?: string | null;
  }>;
  lmCalls: Array<{
    callId: string;
    model: string | null;
    startTime: number;
    endTime: number | null;
    durationMs: number | null;
    iteration: number | null;
    phase: string | null;
    candidateIdx: number | null;
    inputs: Record<string, unknown> | null;
    outputs: Record<string, unknown> | null;
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

  // Calculate average scores from evaluations (same logic as PerformanceTable)
  // Must be called before early returns to follow Rules of Hooks
  const { avgSeedScore, avgBestScore, avgImprovement } = useMemo(() => {
    if (!run) return { avgSeedScore: null, avgBestScore: null, avgImprovement: null };

    // Only use valset evaluations (seed_validation and valset phases)
    // Exclude minibatch_parent and minibatch_new which are exploratory evaluations
    const valsetPhases = new Set(["seed_validation", "valset"]);
    const valsetSet = run.valsetExampleIds ? new Set(run.valsetExampleIds) : null;
    const filteredEvaluations = run.evaluations.filter((ev) => {
      const isValsetPhase = valsetPhases.has(ev.phase);
      const isValsetExample = !valsetSet || valsetSet.has(ev.exampleId);
      return isValsetPhase && isValsetExample;
    });

    // Group evaluations by example_id
    const byExample = new Map<string, typeof filteredEvaluations>();
    for (const ev of filteredEvaluations) {
      const existing = byExample.get(ev.exampleId) || [];
      existing.push(ev);
      byExample.set(ev.exampleId, existing);
    }

    const hasCandidateIdx = run.evaluations.some((e) => e.candidateIdx !== null);
    const seedCandidateIdx = 0;

    const entries: { seedScore: number; bestScore: number }[] = [];

    for (const [, evals] of byExample) {
      let seedEval: typeof evals[0] | undefined;
      let bestEval: typeof evals[0] | undefined;

      if (hasCandidateIdx) {
        // Seed: candidate 0, fallback to earliest by timestamp
        seedEval =
          evals.find((e) => e.candidateIdx === seedCandidateIdx) ??
          evals.reduce((earliest, current) =>
            (current.timestamp ?? 0) < (earliest.timestamp ?? 0) ? current : earliest
          );

        // Best: use bestCandidateIdx if provided (authoritative), otherwise max score
        if (run.bestCandidateIdx != null) {
          // bestCandidateIdx is authoritative - use it even if it equals seed
          bestEval = evals.find((e) => e.candidateIdx === run.bestCandidateIdx);
          // Fallback to max score if best candidate missing for this example
          if (!bestEval) {
            bestEval = evals.reduce((best, current) =>
              current.score > best.score ? current : best
            , evals[0]);
          }
        } else {
          // No bestCandidateIdx provided - use max score per example
          bestEval = evals.reduce((best, current) =>
            current.score > best.score ? current : best
          , evals[0]);
        }
      } else {
        // Fallback: timestamp-based comparison
        const sortedEvals = [...evals].sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0));
        if (sortedEvals.length >= 1) {
          seedEval = sortedEvals[0];
          bestEval = sortedEvals.reduce((best, current) =>
            current.score > best.score ? current : best
          , sortedEvals[0]);
        }
      }

      if (seedEval && bestEval) {
        entries.push({ seedScore: seedEval.score, bestScore: bestEval.score });
      }
    }

    if (entries.length === 0) {
      return { avgSeedScore: null, avgBestScore: null, avgImprovement: null };
    }

    const avgSeed = entries.reduce((sum, e) => sum + e.seedScore, 0) / entries.length;
    const avgBest = entries.reduce((sum, e) => sum + e.bestScore, 0) / entries.length;

    return {
      avgSeedScore: avgSeed,
      avgBestScore: avgBest,
      avgImprovement: avgBest - avgSeed,
    };
  }, [run]);

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
      {/* Tabs */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="logs">Logs</TabsTrigger>
          <TabsTrigger value="iterations">Iterations</TabsTrigger>
          <TabsTrigger value="lineage">Lineage</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6 mt-6">
          {/* Stats */}
          <StatsCards
            iterations={run.totalIterations}
            candidates={run.totalCandidates}
            lmCalls={run.totalLmCalls}
            evaluations={run.totalEvaluations}
            avgSeedScore={avgSeedScore ?? run.seedScore}
            avgBestScore={avgBestScore ?? run.bestScore}
            avgImprovement={avgImprovement}
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
        </TabsContent>

        {/* Logs Tab */}
        <TabsContent value="logs" className="mt-6">
          <LogsTab runId={run.id} isRunning={run.status === "RUNNING"} />
        </TabsContent>

        {/* Iterations Tab */}
        <TabsContent value="iterations" className="mt-6">
          <IterationsTab
            iterations={run.iterations}
            lmCalls={run.lmCalls}
            candidates={run.candidates}
            evaluations={run.evaluations.map(ev => ({
              ...ev,
              iteration: ev.iteration ?? undefined,
            }))}
          />
        </TabsContent>

        {/* Lineage Tab */}
        <TabsContent value="lineage" className="mt-6">
          <LineageTab
            candidates={run.candidates}
            evaluations={run.evaluations}
            bestCandidateIdx={run.bestCandidateIdx}
            paretoPrograms={
              run.iterations.length > 0
                ? run.iterations[run.iterations.length - 1].paretoPrograms
                : null
            }
            valsetExampleIds={run.valsetExampleIds}
            iterations={run.iterations}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
