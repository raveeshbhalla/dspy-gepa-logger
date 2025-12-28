"use client";

import { useState, useMemo } from "react";
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

export type Evaluation = {
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

export type ComparisonEntry = {
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

type PerformanceComparisonTableProps = {
  baseEvaluations: Evaluation[];
  compareEvaluations: Evaluation[];
  baseLabel?: string;
  compareLabel?: string;
};

/**
 * Reusable component for comparing two sets of evaluations.
 *
 * Unlike PerformanceTable which infers seed/best from candidateIdx or timestamps,
 * this component takes two explicit evaluation arrays and matches them by exampleId.
 *
 * Used in:
 * - Iterations tab: Compare parent vs proposed mini-batch evaluations
 * - Lineage tab: Compare any two candidates
 */
export function PerformanceComparisonTable({
  baseEvaluations,
  compareEvaluations,
  baseLabel = "Base",
  compareLabel = "Compare",
}: PerformanceComparisonTableProps) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<"improvements" | "regressions" | "same">("improvements");

  const { improvements, regressions, same } = useMemo(() => {
    // Create lookup maps by exampleId
    const baseByExample = new Map<string, Evaluation>();
    for (const ev of baseEvaluations) {
      baseByExample.set(ev.exampleId, ev);
    }

    const compareByExample = new Map<string, Evaluation>();
    for (const ev of compareEvaluations) {
      compareByExample.set(ev.exampleId, ev);
    }

    const improvements: ComparisonEntry[] = [];
    const regressions: ComparisonEntry[] = [];
    const same: ComparisonEntry[] = [];

    // Get all unique example IDs from both sets
    const allExampleIds = new Set([
      ...baseByExample.keys(),
      ...compareByExample.keys(),
    ]);

    for (const exampleId of allExampleIds) {
      const baseEval = baseByExample.get(exampleId);
      const compareEval = compareByExample.get(exampleId);

      // Skip if we don't have both evaluations
      if (!baseEval || !compareEval) continue;

      const entry: ComparisonEntry = {
        exampleId,
        seedScore: baseEval.score,
        bestScore: compareEval.score,
        delta: compareEval.score - baseEval.score,
        seedFeedback: baseEval.feedback,
        bestFeedback: compareEval.feedback,
        inputs: baseEval.exampleInputs || compareEval.exampleInputs,
        seedPrediction: baseEval.predictionPreview,
        bestPrediction: compareEval.predictionPreview,
        seedPredictionRef: baseEval.predictionRef,
        bestPredictionRef: compareEval.predictionRef,
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
  }, [baseEvaluations, compareEvaluations]);

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

  const handleTabChange = (value: string) => {
    setActiveTab(value as "improvements" | "regressions" | "same");
    setSelectedIndex(null);
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
            <TableHead className="text-right">{baseLabel}</TableHead>
            <TableHead className="text-right">{compareLabel}</TableHead>
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

  // Show empty state if no evaluations to compare
  if (baseEvaluations.length === 0 && compareEvaluations.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-4 text-center">
        No evaluations to compare.
      </p>
    );
  }

  return (
    <>
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
