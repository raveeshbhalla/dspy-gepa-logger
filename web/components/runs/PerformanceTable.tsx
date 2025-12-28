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
  phase?: string;
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
  bestCandidateIdx: number | null;
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
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<"improvements" | "regressions" | "same">("improvements");

  const { improvements, regressions, same } = useMemo(() => {
    // Only use valset evaluations (seed_validation and valset phases)
    // Exclude minibatch_parent and minibatch_new which are exploratory evaluations
    const valsetPhases = new Set(["seed_validation", "valset"]);
    const valsetSet = valsetExampleIds ? new Set(valsetExampleIds) : null;
    const filteredEvaluations = evaluations.filter((ev) => {
      const isValsetPhase = !ev.phase || valsetPhases.has(ev.phase);
      const isValsetExample = !valsetSet || valsetSet.has(ev.exampleId);
      return isValsetPhase && isValsetExample;
    });

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
        // Seed: candidate 0, fallback to earliest by timestamp
        seedEval =
          evals.find((e) => e.candidateIdx === seedCandidateIdx) ??
          evals.reduce((earliest, current) =>
            (current.timestamp ?? 0) < (earliest.timestamp ?? 0) ? current : earliest
          );

        // Best: use bestCandidateIdx if provided (authoritative), otherwise max score
        if (bestCandidateIdx != null) {
          // bestCandidateIdx is authoritative - use it even if it equals seed
          bestEval = evals.find((e) => e.candidateIdx === bestCandidateIdx);
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

    return { improvements, regressions, same };
  }, [evaluations, seedCandidateIdx, bestCandidateIdx, valsetExampleIds]);

  // Get the current entries based on active tab
  const getCurrentEntries = () => {
    switch (activeTab) {
      case "improvements": return improvements;
      case "regressions": return regressions;
      case "same": return same;
    }
  };

  const currentEntries = getCurrentEntries();
  const selectedEntry = selectedIndex !== null ? currentEntries[selectedIndex] : null;

  const handleNavigate = (index: number) => {
    if (index >= 0 && index < currentEntries.length) {
      setSelectedIndex(index);
    }
  };

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
              onClick={() => setSelectedIndex(idx)}
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

  const handleTabChange = (value: string) => {
    setActiveTab(value as "improvements" | "regressions" | "same");
    setSelectedIndex(null); // Reset selection when switching tabs
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Performance Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={handleTabChange}>
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
        entries={currentEntries}
        currentIndex={selectedIndex ?? 0}
        onNavigate={handleNavigate}
        onClose={() => setSelectedIndex(null)}
      />
    </>
  );
}
