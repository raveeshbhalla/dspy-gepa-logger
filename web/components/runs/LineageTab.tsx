"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

type Candidate = {
  candidateIdx: number;
  content: Record<string, string>;
  parentIdx: number | null;
  createdAtIter?: number | null;
};

type Evaluation = {
  evalId: string;
  exampleId: string;
  candidateIdx: number | null;
  score: number;
  feedback: string | null;
  timestamp?: number;
};

type TreeNode = {
  candidateIdx: number;
  content: Record<string, string>;
  parentIdx: number | null;
  children: TreeNode[];
  avgScore: number | null;
  isPareto: boolean;
  createdAtIter: number | null;
};

type LineageTabProps = {
  candidates: Candidate[];
  evaluations: Evaluation[];
  bestCandidateIdx: number | null;
  paretoPrograms: Record<string, number> | null;
  valsetExampleIds: string[] | null;
};

export function LineageTab({
  candidates,
  evaluations,
  bestCandidateIdx,
  paretoPrograms,
  valsetExampleIds,
}: LineageTabProps) {
  const [selectedCandidates, setSelectedCandidates] = useState<number[]>([]);

  // Build tree structure
  const tree = useMemo(() => {
    // Calculate average score for each candidate on valset
    const candidateScores = new Map<number, number[]>();
    const valsetIds = new Set(valsetExampleIds || []);

    // Check if candidateIdx is available
    const hasCandidateIdx = evaluations.some((e) => e.candidateIdx !== null);

    if (hasCandidateIdx) {
      // Use candidateIdx-based grouping
      evaluations.forEach((ev) => {
        if (ev.candidateIdx == null) return;
        if (valsetExampleIds && !valsetIds.has(ev.exampleId)) return;

        if (!candidateScores.has(ev.candidateIdx)) {
          candidateScores.set(ev.candidateIdx, []);
        }
        candidateScores.get(ev.candidateIdx)!.push(ev.score);
      });
    } else {
      // Fallback: estimate scores by iteration
      // Group evaluations by exampleId, find max score per example for each candidate
      // Seed (candidate 0) = first evaluation per example (by timestamp)
      // Other candidates = can't reliably determine, so skip scoring
      // At minimum, calculate seed score from first evaluations
      const byExample = new Map<string, typeof evaluations>();
      evaluations.forEach((ev) => {
        if (valsetExampleIds && !valsetIds.has(ev.exampleId)) return;
        const existing = byExample.get(ev.exampleId) || [];
        existing.push(ev);
        byExample.set(ev.exampleId, existing);
      });

      const seedScores: number[] = [];
      byExample.forEach((evals) => {
        // Sort by timestamp, first = seed evaluation
        const sorted = [...evals].sort((a, b) =>
          ((a as { timestamp?: number }).timestamp ?? 0) - ((b as { timestamp?: number }).timestamp ?? 0)
        );
        if (sorted.length > 0) {
          seedScores.push(sorted[0].score);
        }
      });
      if (seedScores.length > 0) {
        candidateScores.set(0, seedScores);
      }
    }

    const avgScores = new Map<number, number>();
    candidateScores.forEach((scores, idx) => {
      avgScores.set(idx, scores.reduce((a, b) => a + b, 0) / scores.length);
    });

    // Determine which candidates are on pareto
    const paretoSet = new Set<number>();
    if (paretoPrograms) {
      Object.values(paretoPrograms).forEach((idx) => {
        if (typeof idx === "number") {
          paretoSet.add(idx);
        }
      });
    }

    // Build nodes
    const nodes: Map<number, TreeNode> = new Map();
    candidates.forEach((cand) => {
      nodes.set(cand.candidateIdx, {
        candidateIdx: cand.candidateIdx,
        content: cand.content,
        parentIdx: cand.parentIdx,
        children: [],
        avgScore: avgScores.get(cand.candidateIdx) ?? null,
        isPareto: paretoSet.has(cand.candidateIdx),
        createdAtIter: cand.createdAtIter ?? null,
      });
    });

    // Link children to parents
    nodes.forEach((node) => {
      if (node.parentIdx != null && nodes.has(node.parentIdx)) {
        nodes.get(node.parentIdx)!.children.push(node);
      }
    });

    // Find root (candidate 0)
    return nodes.get(0) || null;
  }, [candidates, evaluations, paretoPrograms, valsetExampleIds]);

  function toggleCandidate(idx: number) {
    setSelectedCandidates((prev) => {
      if (prev.includes(idx)) {
        return prev.filter((i) => i !== idx);
      }
      if (prev.length >= 2) {
        return [prev[1], idx];
      }
      return [...prev, idx];
    });
  }

  function getNodeColor(node: TreeNode): string {
    if (node.candidateIdx === bestCandidateIdx) return "bg-green-500";
    if (node.candidateIdx === 0) return "bg-orange-500";
    if (node.isPareto) return "bg-blue-500";
    return "bg-muted-foreground";
  }

  function renderNode(node: TreeNode, depth: number = 0, siblingIndex: number = 0): React.ReactNode {
    const isSelected = selectedCandidates.includes(node.candidateIdx);
    const nodeWidth = 120;
    const nodeHeight = 70;
    const horizontalGap = 30;
    const verticalGap = 40;

    return (
      <div key={node.candidateIdx} className="flex flex-col items-center">
        <div
          onClick={() => toggleCandidate(node.candidateIdx)}
          className={`
            relative cursor-pointer p-3 rounded-lg border-2 transition-all
            ${isSelected ? "border-primary ring-2 ring-primary/30" : "border-border"}
            hover:border-primary/50
            bg-card
          `}
          style={{ minWidth: nodeWidth }}
        >
          <div className="flex items-center gap-2 mb-1">
            <div className={`w-3 h-3 rounded-full ${getNodeColor(node)}`} />
            <span className="font-medium">#{node.candidateIdx}</span>
          </div>
          {node.avgScore != null && (
            <div className="text-sm">
              <span
                className={
                  node.avgScore >= 0.7
                    ? "text-green-500"
                    : node.avgScore >= 0.4
                    ? "text-yellow-500"
                    : "text-red-500"
                }
              >
                {(node.avgScore * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {node.candidateIdx === 0 && (
            <Badge variant="outline" className="text-xs mt-1">
              Seed
            </Badge>
          )}
          {node.candidateIdx === bestCandidateIdx && (
            <Badge variant="default" className="text-xs mt-1">
              Best
            </Badge>
          )}
        </div>

        {/* Children */}
        {node.children.length > 0 && (
          <>
            {/* Vertical connector */}
            <div className="w-px h-8 bg-border" />
            {/* Horizontal bar if multiple children */}
            {node.children.length > 1 && (
              <div
                className="h-px bg-border"
                style={{
                  width: `${(node.children.length - 1) * (nodeWidth + horizontalGap)}px`,
                }}
              />
            )}
            {/* Children row */}
            <div className="flex gap-8">
              {node.children
                .sort((a, b) => a.candidateIdx - b.candidateIdx)
                .map((child, idx) => (
                  <div key={child.candidateIdx} className="flex flex-col items-center">
                    {/* Vertical connector to child */}
                    <div className="w-px h-4 bg-border" />
                    {renderNode(child, depth + 1, idx)}
                  </div>
                ))}
            </div>
          </>
        )}
      </div>
    );
  }

  function renderComparison() {
    if (selectedCandidates.length !== 2) return null;

    const [idx1, idx2] = selectedCandidates.sort((a, b) => a - b);
    const cand1 = candidates.find((c) => c.candidateIdx === idx1);
    const cand2 = candidates.find((c) => c.candidateIdx === idx2);

    if (!cand1 || !cand2) return null;

    const allKeys = new Set([
      ...Object.keys(cand1.content),
      ...Object.keys(cand2.content),
    ]);

    // Get scores for both candidates
    const valsetIds = new Set(valsetExampleIds || []);
    const hasCandidateIdx = evaluations.some((e) => e.candidateIdx !== null);

    let avgScore1: number | null = null;
    let avgScore2: number | null = null;

    if (hasCandidateIdx) {
      // Use candidateIdx-based filtering
      const cand1Evals = evaluations.filter(
        (ev) =>
          ev.candidateIdx === idx1 &&
          (valsetExampleIds ? valsetIds.has(ev.exampleId) : true)
      );
      const cand2Evals = evaluations.filter(
        (ev) =>
          ev.candidateIdx === idx2 &&
          (valsetExampleIds ? valsetIds.has(ev.exampleId) : true)
      );

      avgScore1 =
        cand1Evals.length > 0
          ? cand1Evals.reduce((a, b) => a + b.score, 0) / cand1Evals.length
          : null;
      avgScore2 =
        cand2Evals.length > 0
          ? cand2Evals.reduce((a, b) => a + b.score, 0) / cand2Evals.length
          : null;
    } else {
      // Fallback: timestamp-based scoring
      // Group by exampleId and use timestamp ordering
      const byExample = new Map<string, Evaluation[]>();
      evaluations.forEach((ev) => {
        if (valsetExampleIds && !valsetIds.has(ev.exampleId)) return;
        const existing = byExample.get(ev.exampleId) || [];
        existing.push(ev);
        byExample.set(ev.exampleId, existing);
      });

      // For candidate 0 (seed), use first evaluation per example
      // For other candidates, use best score per example
      if (idx1 === 0) {
        const scores: number[] = [];
        byExample.forEach((evals) => {
          const sorted = [...evals].sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0));
          if (sorted.length > 0) scores.push(sorted[0].score);
        });
        if (scores.length > 0) {
          avgScore1 = scores.reduce((a, b) => a + b, 0) / scores.length;
        }
      }

      // For the other candidate, we can only use best score as fallback
      const bestScores: number[] = [];
      byExample.forEach((evals) => {
        if (evals.length > 0) {
          const best = evals.reduce((max, e) => e.score > max.score ? e : max, evals[0]);
          bestScores.push(best.score);
        }
      });
      if (bestScores.length > 0) {
        avgScore2 = bestScores.reduce((a, b) => a + b, 0) / bestScores.length;
      }
    }

    return (
      <Card className="mt-8">
        <CardHeader>
          <CardTitle className="text-lg">
            Comparing Candidate #{idx1} vs #{idx2}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Score comparison */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-muted/50 p-4 rounded-lg text-center">
              <p className="text-sm text-muted-foreground">Candidate #{idx1}</p>
              <p className="text-2xl font-bold">
                {avgScore1 != null ? `${(avgScore1 * 100).toFixed(1)}%` : "-"}
              </p>
            </div>
            <div className="bg-muted/50 p-4 rounded-lg text-center">
              <p className="text-sm text-muted-foreground">Delta</p>
              <p
                className={`text-2xl font-bold ${
                  avgScore1 != null && avgScore2 != null
                    ? avgScore2 > avgScore1
                      ? "text-green-500"
                      : avgScore2 < avgScore1
                      ? "text-red-500"
                      : "text-muted-foreground"
                    : ""
                }`}
              >
                {avgScore1 != null && avgScore2 != null
                  ? `${avgScore2 > avgScore1 ? "+" : ""}${((avgScore2 - avgScore1) * 100).toFixed(1)}%`
                  : "-"}
              </p>
            </div>
            <div className="bg-muted/50 p-4 rounded-lg text-center">
              <p className="text-sm text-muted-foreground">Candidate #{idx2}</p>
              <p className="text-2xl font-bold">
                {avgScore2 != null ? `${(avgScore2 * 100).toFixed(1)}%` : "-"}
              </p>
            </div>
          </div>

          {/* Prompt comparison */}
          <div>
            <h4 className="font-medium mb-3">Prompt Comparison</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h5 className="text-sm text-muted-foreground mb-2">
                  Candidate #{idx1}
                </h5>
                <div className="space-y-2">
                  {Array.from(allKeys).map((key) => {
                    const value = cand1.content[key] || "(empty)";
                    const changed = cand1.content[key] !== cand2.content[key];
                    return (
                      <div
                        key={key}
                        className={`p-2 rounded text-sm ${
                          changed
                            ? "bg-red-500/10 border border-red-500/20"
                            : "bg-muted/50"
                        }`}
                      >
                        <span className="font-medium text-muted-foreground">
                          {key}:
                        </span>
                        <p className="whitespace-pre-wrap mt-1">{value}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
              <div>
                <h5 className="text-sm text-muted-foreground mb-2">
                  Candidate #{idx2}
                </h5>
                <div className="space-y-2">
                  {Array.from(allKeys).map((key) => {
                    const value = cand2.content[key] || "(empty)";
                    const changed = cand1.content[key] !== cand2.content[key];
                    return (
                      <div
                        key={key}
                        className={`p-2 rounded text-sm ${
                          changed
                            ? "bg-green-500/10 border border-green-500/20"
                            : "bg-muted/50"
                        }`}
                      >
                        <span className="font-medium text-muted-foreground">
                          {key}:
                        </span>
                        <p className="whitespace-pre-wrap mt-1">{value}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          <Button
            variant="outline"
            onClick={() => setSelectedCandidates([])}
            className="mt-4"
          >
            Clear Selection
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (candidates.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-muted-foreground">
          <p>No candidates yet.</p>
          <p className="text-sm mt-2">
            Candidates will appear here as the optimization runs.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Candidate Lineage Tree</CardTitle>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-orange-500" />
                <span className="text-muted-foreground">Seed</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <span className="text-muted-foreground">Best</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-500" />
                <span className="text-muted-foreground">Pareto</span>
              </div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            Click on two candidates to compare them.
            {selectedCandidates.length === 1 && " Select one more to compare."}
          </p>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <div className="flex justify-center p-4 min-w-fit">
            {tree ? renderNode(tree) : <p>No tree to display</p>}
          </div>
        </CardContent>
      </Card>

      {renderComparison()}
    </div>
  );
}
