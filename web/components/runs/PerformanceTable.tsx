"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { EvaluationModal } from "./EvaluationModal";

type Evaluation = {
  evalId: string;
  exampleId: string;
  candidateIdx: number | null;
  score: number;
  feedback: string | null;
  exampleInputs: Record<string, unknown> | null;
  predictionPreview: string | null;
  predictionRef: Record<string, unknown> | null;
  timestamp?: number;
};

type PerformanceTableProps = {
  evaluations: Evaluation[];
  seedCandidateIdx: number;
  bestCandidateIdx: number;
  valsetExampleIds?: string[] | null;
};

type ComparisonEntry = {
  exampleId: string;
  seedScore: number;
  bestScore: number;
  delta: number;
  seedFeedback: string | null;
  bestFeedback: string | null;
  inputs: Record<string, unknown> | null;
  seedPrediction: string | null;
  bestPrediction: string | null;
  seedPredictionRef: Record<string, unknown> | null;
  bestPredictionRef: Record<string, unknown> | null;
};

export function PerformanceTable({
  evaluations,
  seedCandidateIdx,
  bestCandidateIdx,
  valsetExampleIds,
}: PerformanceTableProps) {
  const [selectedEntry, setSelectedEntry] = useState<ComparisonEntry | null>(null);

  const { improvements, regressions, same, avgSeedScore, avgBestScore, avgDelta } = useMemo(() => {
    // Filter to validation set only if valsetExampleIds is provided
    const valsetSet = valsetExampleIds ? new Set(valsetExampleIds) : null;
    const filteredEvaluations = valsetSet
      ? evaluations.filter((ev) => valsetSet.has(ev.exampleId))
      : evaluations;

    // Group evaluations by example_id
    const byExample = new Map<string, Evaluation[]>();
    for (const ev of filteredEvaluations) {
      const existing = byExample.get(ev.exampleId) || [];
      existing.push(ev);
      byExample.set(ev.exampleId, existing);
    }

    const improvements: ComparisonEntry[] = [];
    const regressions: ComparisonEntry[] = [];
    const same: ComparisonEntry[] = [];

    // Check if candidateIdx is available (not all null)
    const hasCandidateIdx = evaluations.some((e) => e.candidateIdx !== null);

    for (const [exampleId, evals] of byExample) {
      let seedEval: Evaluation | undefined;
      let bestEval: Evaluation | undefined;

      if (hasCandidateIdx) {
        // Use candidateIdx-based matching
        seedEval = evals.find((e) => e.candidateIdx === seedCandidateIdx);
        bestEval = evals.find((e) => e.candidateIdx === bestCandidateIdx);
      } else {
        // Fallback: timestamp-based comparison (like Python tracker)
        // GEPA doesn't expose candidate_idx through public hooks, so we use
        // timestamp ordering: first eval = seed, best score = optimized
        const sortedEvals = [...evals].sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0));

        if (sortedEvals.length >= 1) {
          // First evaluation (by timestamp) is the seed/baseline
          seedEval = sortedEvals[0];

          // Best score among all evaluations represents the optimized result
          // This matches the Python tracker's logic in get_evaluation_comparison()
          bestEval = sortedEvals.reduce((best, current) =>
            current.score > best.score ? current : best
          , sortedEvals[0]);
        }
      }

      if (!seedEval || !bestEval) continue;

      const entry: ComparisonEntry = {
        exampleId,
        seedScore: seedEval.score,
        bestScore: bestEval.score,
        delta: bestEval.score - seedEval.score,
        seedFeedback: seedEval.feedback,
        bestFeedback: bestEval.feedback,
        inputs: seedEval.exampleInputs || bestEval.exampleInputs,
        seedPrediction: seedEval.predictionPreview,
        bestPrediction: bestEval.predictionPreview,
        seedPredictionRef: seedEval.predictionRef,
        bestPredictionRef: bestEval.predictionRef,
      };

      if (entry.delta > 0.001) {
        improvements.push(entry);
      } else if (entry.delta < -0.001) {
        regressions.push(entry);
      } else {
        same.push(entry);
      }
    }

    // Sort by absolute delta
    improvements.sort((a, b) => b.delta - a.delta);
    regressions.sort((a, b) => a.delta - b.delta);

    // Calculate overall averages
    const allEntries = [...improvements, ...regressions, ...same];
    const totalEntries = allEntries.length;
    const avgSeedScore = totalEntries > 0
      ? allEntries.reduce((sum, e) => sum + e.seedScore, 0) / totalEntries
      : 0;
    const avgBestScore = totalEntries > 0
      ? allEntries.reduce((sum, e) => sum + e.bestScore, 0) / totalEntries
      : 0;
    const avgDelta = avgBestScore - avgSeedScore;

    return { improvements, regressions, same, avgSeedScore, avgBestScore, avgDelta };
  }, [evaluations, seedCandidateIdx, bestCandidateIdx, valsetExampleIds]);

  const renderTable = (entries: ComparisonEntry[], type: "improve" | "regress" | "same") => {
    if (entries.length === 0) {
      return (
        <p className="text-sm text-muted-foreground py-4 text-center">
          No {type === "improve" ? "improvements" : type === "regress" ? "regressions" : "unchanged examples"}
        </p>
      );
    }

    // Get input field names from first entry
    const inputFields = entries[0]?.inputs
      ? Object.keys(entries[0].inputs)
      : [];

    return (
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-12">#</TableHead>
            {inputFields.slice(0, 2).map((field) => (
              <TableHead key={field}>{field}</TableHead>
            ))}
            <TableHead className="text-right">Delta</TableHead>
            <TableHead className="text-right">Seed</TableHead>
            <TableHead className="text-right">Best</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {entries.map((entry, idx) => (
            <TableRow
              key={entry.exampleId}
              className="cursor-pointer hover:bg-accent"
              onClick={() => setSelectedEntry(entry)}
            >
              <TableCell className="font-mono text-xs text-muted-foreground">
                {idx + 1}
              </TableCell>
              {inputFields.slice(0, 2).map((field) => (
                <TableCell key={field} className="max-w-[200px] truncate text-sm">
                  {String(entry.inputs?.[field] || "").slice(0, 50)}
                </TableCell>
              ))}
              <TableCell
                className={`text-right font-mono text-sm font-semibold ${
                  entry.delta > 0
                    ? "text-green-500"
                    : entry.delta < 0
                    ? "text-red-500"
                    : ""
                }`}
              >
                {entry.delta > 0 ? "+" : ""}
                {(entry.delta * 100).toFixed(0)}%
              </TableCell>
              <TableCell className="text-right font-mono text-sm text-muted-foreground">
                {(entry.seedScore * 100).toFixed(0)}%
              </TableCell>
              <TableCell className="text-right font-mono text-sm text-muted-foreground">
                {(entry.bestScore * 100).toFixed(0)}%
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Performance Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          {/* Overall Average Performance Summary */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <p className="text-2xl font-semibold">
                {(avgSeedScore * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground">Avg Seed Score</p>
            </div>
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <p className="text-2xl font-semibold">
                {(avgBestScore * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground">Avg Best Score</p>
            </div>
            <div
              className={`text-center p-3 rounded-lg ${
                avgDelta > 0
                  ? "bg-green-500/10"
                  : avgDelta < 0
                  ? "bg-red-500/10"
                  : "bg-muted/50"
              }`}
            >
              <p
                className={`text-2xl font-semibold ${
                  avgDelta > 0
                    ? "text-green-500"
                    : avgDelta < 0
                    ? "text-red-500"
                    : ""
                }`}
              >
                {avgDelta > 0 ? "+" : ""}
                {(avgDelta * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground">Avg Improvement</p>
            </div>
          </div>

          <Tabs defaultValue="improvements">
            <TabsList className="mb-4">
              <TabsTrigger value="improvements" className="gap-2">
                Improvements
                <span className="text-xs bg-green-500/20 text-green-500 px-1.5 py-0.5 rounded">
                  {improvements.length}
                </span>
              </TabsTrigger>
              <TabsTrigger value="regressions" className="gap-2">
                Regressions
                <span className="text-xs bg-red-500/20 text-red-500 px-1.5 py-0.5 rounded">
                  {regressions.length}
                </span>
              </TabsTrigger>
              <TabsTrigger value="same" className="gap-2">
                Unchanged
                <span className="text-xs bg-muted px-1.5 py-0.5 rounded">
                  {same.length}
                </span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="improvements">
              {renderTable(improvements, "improve")}
            </TabsContent>

            <TabsContent value="regressions">
              {renderTable(regressions, "regress")}
            </TabsContent>

            <TabsContent value="same">
              {renderTable(same, "same")}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      <EvaluationModal
        entry={selectedEntry}
        onClose={() => setSelectedEntry(null)}
      />
    </>
  );
}
