"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

type Iteration = {
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
};

type LmCall = {
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
};

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
  iteration?: number | null;
  score: number;
  feedback: string | null;
};

type IterationsTabProps = {
  iterations: Iteration[];
  lmCalls: LmCall[];
  candidates: Candidate[];
  evaluations: Evaluation[];
};

export function IterationsTab({
  iterations,
  lmCalls,
  candidates,
  evaluations,
}: IterationsTabProps) {
  const [expandedIteration, setExpandedIteration] = useState<number | null>(null);

  function getIterationLmCalls(iterNum: number): LmCall[] {
    return lmCalls.filter((lm) => lm.iteration === iterNum);
  }

  function getReflectionCalls(iterNum: number): LmCall[] {
    return lmCalls.filter(
      (lm) => lm.iteration === iterNum && lm.phase === "reflection"
    );
  }

  function getProposalCalls(iterNum: number): LmCall[] {
    return lmCalls.filter(
      (lm) => lm.iteration === iterNum && lm.phase === "proposal"
    );
  }

  function getIterationEvaluations(iterNum: number): Evaluation[] {
    return evaluations.filter((ev) => ev.iteration === iterNum);
  }

  function getCandidatesCreatedAt(iterNum: number): Candidate[] {
    return candidates.filter((c) => c.createdAtIter === iterNum);
  }

  function getCandidate(idx: number): Candidate | undefined {
    return candidates.find((c) => c.candidateIdx === idx);
  }

  function formatTimestamp(timestamp: number): string {
    return new Date(timestamp * 1000).toLocaleTimeString("en-US", {
      hour12: false,
    });
  }

  function renderPromptComparison(
    parentPrompt: Record<string, string> | undefined,
    childPrompt: Record<string, string> | undefined,
    parentLabel: string,
    childLabel: string
  ) {
    if (!parentPrompt && !childPrompt) return null;

    const allKeys = new Set([
      ...Object.keys(parentPrompt || {}),
      ...Object.keys(childPrompt || {}),
    ]);

    return (
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h4 className="font-medium text-sm mb-2 text-muted-foreground">
            {parentLabel}
          </h4>
          <div className="space-y-2">
            {Array.from(allKeys).map((key) => {
              const value = parentPrompt?.[key] || "(empty)";
              const changed = parentPrompt?.[key] !== childPrompt?.[key];
              return (
                <div
                  key={key}
                  className={`p-2 rounded text-sm ${
                    changed ? "bg-red-500/10 border border-red-500/20" : "bg-muted/50"
                  }`}
                >
                  <span className="font-medium text-muted-foreground">{key}:</span>
                  <p className="whitespace-pre-wrap mt-1">{value}</p>
                </div>
              );
            })}
          </div>
        </div>
        <div>
          <h4 className="font-medium text-sm mb-2 text-muted-foreground">
            {childLabel}
          </h4>
          <div className="space-y-2">
            {Array.from(allKeys).map((key) => {
              const value = childPrompt?.[key] || "(empty)";
              const changed = parentPrompt?.[key] !== childPrompt?.[key];
              return (
                <div
                  key={key}
                  className={`p-2 rounded text-sm ${
                    changed ? "bg-green-500/10 border border-green-500/20" : "bg-muted/50"
                  }`}
                >
                  <span className="font-medium text-muted-foreground">{key}:</span>
                  <p className="whitespace-pre-wrap mt-1">{value}</p>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  function renderLmCallDetails(call: LmCall) {
    return (
      <div className="border rounded-lg p-4 space-y-3 bg-muted/30">
        <div className="flex items-center gap-2 flex-wrap">
          <Badge variant="outline">{call.model || "unknown"}</Badge>
          <Badge variant="secondary">{call.phase}</Badge>
          {call.durationMs && (
            <span className="text-xs text-muted-foreground">
              {call.durationMs.toFixed(0)}ms
            </span>
          )}
        </div>

        <details className="text-sm">
          <summary className="cursor-pointer font-medium text-muted-foreground hover:text-foreground">
            Inputs
          </summary>
          <pre className="mt-2 p-3 bg-muted rounded text-xs overflow-x-auto whitespace-pre-wrap max-h-64 overflow-y-auto">
            {JSON.stringify(call.inputs, null, 2)}
          </pre>
        </details>

        <details className="text-sm">
          <summary className="cursor-pointer font-medium text-muted-foreground hover:text-foreground">
            Outputs
          </summary>
          <pre className="mt-2 p-3 bg-muted rounded text-xs overflow-x-auto whitespace-pre-wrap max-h-64 overflow-y-auto">
            {JSON.stringify(call.outputs, null, 2)}
          </pre>
        </details>
      </div>
    );
  }

  if (iterations.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-muted-foreground">
          <p>No iterations yet.</p>
          <p className="text-sm mt-2">
            Iterations will appear here as the optimization runs.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {iterations.map((iter) => {
        const isExpanded = expandedIteration === iter.iterationNumber;
        const reflectionCalls = getReflectionCalls(iter.iterationNumber);
        const proposalCalls = getProposalCalls(iter.iterationNumber);
        const iterEvals = getIterationEvaluations(iter.iterationNumber);
        const newCandidates = getCandidatesCreatedAt(iter.iterationNumber);
        const parentCandidate = iter.parentCandidateIdx != null
          ? getCandidate(iter.parentCandidateIdx)
          : undefined;

        return (
          <Card key={iter.iterationNumber}>
            <CardHeader
              className="cursor-pointer hover:bg-muted/50 transition-colors"
              onClick={() =>
                setExpandedIteration(isExpanded ? null : iter.iterationNumber)
              }
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <CardTitle className="text-lg">
                    Iteration {iter.iterationNumber}
                  </CardTitle>
                  <Badge variant="secondary">{iter.totalEvals} evals</Badge>
                  <Badge variant="outline">
                    {iter.numCandidates} candidates
                  </Badge>
                  <Badge variant="outline">
                    Pareto: {iter.paretoSize}
                  </Badge>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">
                    {formatTimestamp(iter.timestamp)}
                  </span>
                  <Button variant="ghost" size="sm">
                    {isExpanded ? "Collapse" : "Expand"}
                  </Button>
                </div>
              </div>
            </CardHeader>

            {isExpanded && (
              <CardContent className="space-y-6">
                <Tabs defaultValue="prompts">
                  <TabsList>
                    <TabsTrigger value="prompts">Prompt Changes</TabsTrigger>
                    <TabsTrigger value="reflection">
                      Reflection ({reflectionCalls.length})
                    </TabsTrigger>
                    <TabsTrigger value="proposal">
                      Proposal ({proposalCalls.length})
                    </TabsTrigger>
                    <TabsTrigger value="evals">
                      Evaluations ({iterEvals.length})
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="prompts" className="mt-4">
                    {newCandidates.length > 0 ? (
                      <div className="space-y-6">
                        {newCandidates.map((child) => {
                          const parent = child.parentIdx != null
                            ? getCandidate(child.parentIdx)
                            : undefined;
                          return (
                            <div key={child.candidateIdx}>
                              <h4 className="font-medium mb-3">
                                Candidate #{child.candidateIdx}
                                {parent && (
                                  <span className="text-muted-foreground font-normal">
                                    {" "}(mutated from #{parent.candidateIdx})
                                  </span>
                                )}
                              </h4>
                              {renderPromptComparison(
                                parent?.content,
                                child.content,
                                `Parent (#${parent?.candidateIdx ?? "?"})`,
                                `Child (#${child.candidateIdx})`
                              )}
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="text-muted-foreground text-center py-8">
                        No new candidates created in this iteration.
                      </p>
                    )}
                  </TabsContent>

                  <TabsContent value="reflection" className="mt-4 space-y-4">
                    {reflectionCalls.length > 0 ? (
                      reflectionCalls.map((call) => (
                        <div key={call.callId}>
                          {renderLmCallDetails(call)}
                        </div>
                      ))
                    ) : (
                      <p className="text-muted-foreground text-center py-8">
                        No reflection LM calls recorded for this iteration.
                      </p>
                    )}
                  </TabsContent>

                  <TabsContent value="proposal" className="mt-4 space-y-4">
                    {proposalCalls.length > 0 ? (
                      proposalCalls.map((call) => (
                        <div key={call.callId}>
                          {renderLmCallDetails(call)}
                        </div>
                      ))
                    ) : (
                      <p className="text-muted-foreground text-center py-8">
                        No proposal LM calls recorded for this iteration.
                      </p>
                    )}
                  </TabsContent>

                  <TabsContent value="evals" className="mt-4">
                    {iterEvals.length > 0 ? (
                      <div className="space-y-2">
                        <div className="grid grid-cols-4 gap-4 text-sm font-medium text-muted-foreground border-b pb-2">
                          <span>Example ID</span>
                          <span>Candidate</span>
                          <span>Score</span>
                          <span>Feedback</span>
                        </div>
                        {iterEvals.slice(0, 20).map((ev) => (
                          <div
                            key={ev.evalId}
                            className="grid grid-cols-4 gap-4 text-sm py-2 border-b border-muted"
                          >
                            <span className="font-mono text-xs truncate">
                              {ev.exampleId.slice(0, 12)}...
                            </span>
                            <span>#{ev.candidateIdx ?? "?"}</span>
                            <span
                              className={
                                ev.score >= 0.7
                                  ? "text-green-500"
                                  : ev.score >= 0.4
                                  ? "text-yellow-500"
                                  : "text-red-500"
                              }
                            >
                              {(ev.score * 100).toFixed(1)}%
                            </span>
                            <span className="text-muted-foreground truncate">
                              {ev.feedback || "-"}
                            </span>
                          </div>
                        ))}
                        {iterEvals.length > 20 && (
                          <p className="text-sm text-muted-foreground text-center pt-2">
                            ... and {iterEvals.length - 20} more evaluations
                          </p>
                        )}
                      </div>
                    ) : (
                      <p className="text-muted-foreground text-center py-8">
                        No evaluations recorded for this iteration.
                      </p>
                    )}
                  </TabsContent>
                </Tabs>
              </CardContent>
            )}
          </Card>
        );
      })}
    </div>
  );
}
