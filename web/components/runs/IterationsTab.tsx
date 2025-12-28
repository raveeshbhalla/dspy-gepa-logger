"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PerformanceComparisonTable, type Evaluation as FullEvaluation } from "./PerformanceComparisonTable";

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
  phase?: string | null;
  score: number;
  feedback: string | null;
  exampleInputs: Record<string, unknown> | null;
  predictionPreview: string | null;
  predictionRef: Record<string, unknown> | null;
  timestamp?: number;
};

type IterationsTabProps = {
  iterations: Iteration[];
  lmCalls: LmCall[];
  candidates: Candidate[];
  evaluations: Evaluation[];
  allIterations?: Iteration[]; // All iterations for checking pareto membership
};

export function IterationsTab({
  iterations,
  lmCalls,
  candidates,
  evaluations,
  allIterations,
}: IterationsTabProps) {
  const [expandedIteration, setExpandedIteration] = useState<number | null>(null);

  function getReflectionCalls(iterNum: number): LmCall[] {
    // Due to context timing in GEPA, reflection LM calls made during iteration N
    // are logged with iteration = N-1 (context is set AFTER state callback).
    // So for iteration N (N > 0), we look for reflection calls with iteration = N-1.
    // For iteration 0, there are no reflection calls (it's the seed evaluation).
    const targetIteration = iterNum > 0 ? iterNum - 1 : iterNum;
    return lmCalls.filter(
      (lm) => lm.iteration === targetIteration && lm.phase === "reflection"
    );
  }

  /**
   * Split iteration evaluations into parent vs proposed sets.
   * Within each iteration, GEPA evaluates parent first, then proposed.
   *
   * NOTE on GEPA evaluation tagging (gepa-observable):
   * - Both parent and proposed evals have the same iteration number
   * - They are distinguished by phase: "minibatch_parent" vs "minibatch_new"
   *
   * Fallback for legacy behavior (dspy_gepa_logger):
   * - Proposed evaluations have iteration = iterNum - 1
   * - Parent evaluations have iteration = NULL
   */
  function getIterationComparison(iterNum: number): {
    parentEvals: FullEvaluation[];
    proposedEvals: FullEvaluation[];
  } {
    // Strategy 1: Try phase-based matching (gepa-observable)
    // Parent evals have phase="minibatch_parent", proposed have phase="minibatch_new"
    const parentByPhase = evaluations.filter(
      (ev) => ev.iteration === iterNum && ev.phase === "minibatch_parent"
    ) as FullEvaluation[];

    const proposedByPhase = evaluations.filter(
      (ev) => ev.iteration === iterNum && ev.phase === "minibatch_new"
    ) as FullEvaluation[];

    if (parentByPhase.length > 0 || proposedByPhase.length > 0) {
      return { parentEvals: parentByPhase, proposedEvals: proposedByPhase };
    }

    // Strategy 2: Fallback to legacy iteration-based matching
    // Proposed evaluations have iteration = iterNum - 1
    const targetIteration = iterNum - 1;
    const proposedEvals = evaluations.filter(
      (ev) => ev.iteration === targetIteration
    ) as FullEvaluation[];

    if (proposedEvals.length === 0) {
      return { parentEvals: [], proposedEvals: [] };
    }

    // Get the exampleIds and min timestamp from proposed evals
    const proposedExampleIds = new Set(proposedEvals.map((ev) => ev.exampleId));
    const minProposedTimestamp = Math.min(
      ...proposedEvals.map((ev) => ev.timestamp ?? Infinity)
    );

    // Find parent evals: iteration=NULL, same exampleIds, timestamp before proposed
    // Also filter to only get the most recent NULL eval per exampleId before proposed
    const candidateParentEvals = evaluations.filter(
      (ev) =>
        (ev.iteration === null || ev.iteration === undefined) &&
        proposedExampleIds.has(ev.exampleId) &&
        (ev.timestamp ?? 0) < minProposedTimestamp
    );

    // Group by exampleId and take the most recent (closest to proposed timestamp)
    const parentByExample = new Map<string, Evaluation>();
    for (const ev of candidateParentEvals) {
      const existing = parentByExample.get(ev.exampleId);
      if (!existing || (ev.timestamp ?? 0) > (existing.timestamp ?? 0)) {
        parentByExample.set(ev.exampleId, ev);
      }
    }

    const parentEvals = Array.from(parentByExample.values()) as FullEvaluation[];

    return { parentEvals, proposedEvals };
  }

  function getCandidatesCreatedAt(iterNum: number): Candidate[] {
    return candidates.filter((c) => c.createdAtIter === iterNum);
  }

  function getCandidate(idx: number): Candidate | undefined {
    return candidates.find((c) => c.candidateIdx === idx);
  }

  /**
   * Check if a candidate became part of the pareto frontier.
   * Checks the LATEST iteration's paretoPrograms (final state) since that
   * represents which candidates ended up on the frontier.
   * Also checks all iterations in case the format varies.
   */
  function isCandidateInPareto(candidateIdx: number): boolean {
    const iters = allIterations || iterations;

    // Check all iterations (latest first for efficiency)
    const sortedIters = [...iters].sort((a, b) => b.iterationNumber - a.iterationNumber);

    for (const iter of sortedIters) {
      if (iter.paretoPrograms) {
        const paretoIndices = Object.values(iter.paretoPrograms);
        // paretoPrograms can be number or array of numbers depending on backend
        for (const val of paretoIndices) {
          if (Array.isArray(val)) {
            if (val.includes(candidateIdx)) return true;
          } else if (val === candidateIdx) {
            return true;
          }
        }
      }
    }
    return false;
  }

  /**
   * Check if any candidate created in this iteration became a pareto candidate.
   * Uses multiple strategies to determine success.
   */
  function didIterationProduceSuccessfulCandidate(iterNum: number): boolean {
    const iter = iterations.find((i) => i.iterationNumber === iterNum);
    if (!iter) return false;

    // Method 1: Check candidates created at this iteration (from candidates table)
    const createdCandidates = getCandidatesCreatedAt(iterNum);
    if (createdCandidates.some((c) => isCandidateInPareto(c.candidateIdx))) {
      return true;
    }

    // Method 2: Check childCandidateIdxs from iteration data
    if (iter.childCandidateIdxs) {
      try {
        const childIdxs = JSON.parse(iter.childCandidateIdxs) as number[];
        if (childIdxs.some((idx) => isCandidateInPareto(idx))) {
          return true;
        }
      } catch {
        // Invalid JSON, ignore
      }
    }

    // Method 3: Check if pareto size increased from previous iteration
    // This indicates a successful candidate was added
    const prevIter = iterations.find((i) => i.iterationNumber === iterNum - 1);
    const prevParetoSize = prevIter?.paretoSize ?? 1; // Default to 1 (seed)
    if (iter.paretoSize > prevParetoSize) {
      return true;
    }

    // Method 4: Check if this iteration's paretoPrograms contains any non-seed candidates
    // that were created at this iteration or have this iteration as their createdAtIter
    if (iter.paretoPrograms) {
      const paretoIndices = Object.values(iter.paretoPrograms);
      const allParetoIdxs = new Set<number>();
      for (const val of paretoIndices) {
        if (Array.isArray(val)) {
          val.forEach((idx) => allParetoIdxs.add(idx));
        } else if (typeof val === 'number') {
          allParetoIdxs.add(val);
        }
      }

      // Check if any pareto candidate (other than seed) was created at this iteration
      for (const idx of allParetoIdxs) {
        if (idx === 0) continue; // Skip seed
        const candidate = candidates.find((c) => c.candidateIdx === idx);
        if (candidate?.createdAtIter === iterNum) {
          return true;
        }
      }
    }

    return false;
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

  // Filter out iteration 0 (seed validation) - we only show actual optimization iterations
  const displayIterations = iterations.filter((iter) => iter.iterationNumber !== 0);

  if (displayIterations.length === 0) {
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
      {displayIterations.map((iter) => {
        const isExpanded = expandedIteration === iter.iterationNumber;
        const reflectionCalls = getReflectionCalls(iter.iterationNumber);
        const { parentEvals, proposedEvals } = getIterationComparison(iter.iterationNumber);
        const newCandidates = getCandidatesCreatedAt(iter.iterationNumber);
        // Try to get parent candidate from iteration data, or infer from other sources
        let parentCandidate = iter.parentCandidateIdx != null
          ? getCandidate(iter.parentCandidateIdx)
          : undefined;
        // Fallback 1: infer parent from parent evaluations' candidateIdx
        if (!parentCandidate && parentEvals.length > 0) {
          const parentCandidateIdx = parentEvals[0].candidateIdx;
          if (parentCandidateIdx != null) {
            parentCandidate = getCandidate(parentCandidateIdx);
          }
        }
        // Fallback 2: infer parent from reflection calls' candidateIdx
        if (!parentCandidate && reflectionCalls.length > 0) {
          const candidateIdx = reflectionCalls[0].candidateIdx;
          if (candidateIdx != null) {
            parentCandidate = getCandidate(candidateIdx);
          }
        }
        const iterationSuccessful = didIterationProduceSuccessfulCandidate(iter.iterationNumber);

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
                    Iteration {iter.iterationNumber}{" "}
                    <span title={iterationSuccessful ? "Produced a pareto candidate" : "Did not produce a pareto candidate"}>
                      {iterationSuccessful ? "✅" : "❌"}
                    </span>
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
                      Reflection ({(iter.reflectionInput || iter.reflectionOutput) ? 1 : reflectionCalls.length})
                    </TabsTrigger>
                    <TabsTrigger value="evals">
                      Performance Comparison ({Math.min(parentEvals.length, proposedEvals.length)})
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
                    ) : (() => {
                      // Try to extract proposals from: 1) proposedChanges field, 2) reflection call outputs
                      let proposals: Array<Record<string, string>> = [];

                      // First try proposedChanges field
                      if (iter.proposedChanges) {
                        try {
                          const parsed = JSON.parse(iter.proposedChanges);
                          // Handle gepa-observable format: [{"component": "name", "newText": "..."}]
                          // Convert to instruction format: {"name": "..."}
                          if (Array.isArray(parsed)) {
                            const converted: Record<string, string> = {};
                            for (const item of parsed) {
                              if (item.component && item.newText) {
                                converted[item.component] = item.newText;
                              } else if (typeof item === 'object') {
                                // Already in {name: text} format
                                Object.assign(converted, item);
                              }
                            }
                            if (Object.keys(converted).length > 0) {
                              proposals.push(converted);
                            }
                          } else if (typeof parsed === 'object') {
                            proposals.push(parsed);
                          }
                        } catch {
                          // Invalid JSON, will fall through to reflection output
                        }
                      }

                      // If no proposals from proposedChanges, try to extract from reflection call outputs
                      if (proposals.length === 0 && reflectionCalls.length > 0) {
                        for (const call of reflectionCalls) {
                          if (call.outputs) {
                            const outputs = call.outputs as Record<string, unknown>;

                            // Determine the field name from parent candidate
                            let fieldName = 'instructions'; // default
                            if (parentCandidate?.content) {
                              const keys = Object.keys(parentCandidate.content);
                              if (keys.length === 1) {
                                fieldName = keys[0];
                              }
                            }

                            // Strategy 1: Look for common DSPy output field names
                            const proposedInstruction =
                              outputs.proposed_instruction ||
                              outputs.proposedInstruction ||
                              outputs.new_instruction ||
                              outputs.instruction;

                            if (typeof proposedInstruction === 'string' && proposedInstruction.length > 0) {
                              proposals.push({ [fieldName]: proposedInstruction });
                              continue;
                            }

                            // Strategy 2: Try as object with multiple fields
                            const proposedInstructions =
                              outputs.proposed_instructions ||
                              outputs.proposedInstructions ||
                              outputs.new_instructions;

                            if (proposedInstructions && typeof proposedInstructions === 'object') {
                              proposals.push(proposedInstructions as Record<string, string>);
                              continue;
                            }

                            // Strategy 3: Extract string fields that look like instructions
                            // DSPy outputs have the instruction in a field matching the signature output name
                            const extractedProposal: Record<string, string> = {};
                            for (const [key, value] of Object.entries(outputs)) {
                              // Skip metadata/reasoning fields
                              const lowerKey = key.toLowerCase();
                              if (
                                typeof value === 'string' &&
                                value.length > 20 && // Instructions are typically longer
                                !['rationale', 'reasoning', 'analysis', 'explanation'].includes(lowerKey)
                              ) {
                                // Use the parent's field name if there's only one instruction field
                                extractedProposal[fieldName] = value;
                                break; // Take the first substantial string field
                              }
                            }
                            if (Object.keys(extractedProposal).length > 0) {
                              proposals.push(extractedProposal);
                            }
                          }
                        }
                      }

                      if (proposals.length === 0 || !parentCandidate) {
                        return (
                          <p className="text-muted-foreground text-center py-8">
                            No prompt changes in this iteration.
                          </p>
                        );
                      }

                      return (
                        <div className="space-y-6">
                          {proposals.map((proposedContent, idx) => (
                            <div key={idx}>
                              <h4 className="font-medium mb-3">
                                Proposed Change {proposals.length > 1 ? `#${idx + 1}` : ""}
                                <span className="text-muted-foreground font-normal">
                                  {" "}(rejected - mutated from #{parentCandidate.candidateIdx})
                                </span>
                              </h4>
                              {renderPromptComparison(
                                parentCandidate.content,
                                proposedContent,
                                `Parent (#${parentCandidate.candidateIdx})`,
                                "Proposed (rejected)"
                              )}
                            </div>
                          ))}
                        </div>
                      );
                    })()}
                  </TabsContent>

                  <TabsContent value="reflection" className="mt-4 space-y-4">
                    {(() => {
                      // Try to get reflection data from iteration fields first (gepa-observable)
                      const hasIterationReflection = iter.reflectionInput || iter.reflectionOutput;

                      if (hasIterationReflection) {
                        let reflectionInputStr = "";
                        let reflectionOutputStr = "";

                        try {
                          if (iter.reflectionInput) {
                            const parsed = JSON.parse(iter.reflectionInput);
                            reflectionInputStr = JSON.stringify(parsed, null, 2);
                          }
                        } catch {
                          reflectionInputStr = iter.reflectionInput || "";
                        }

                        try {
                          if (iter.reflectionOutput) {
                            const parsed = JSON.parse(iter.reflectionOutput);
                            reflectionOutputStr = JSON.stringify(parsed, null, 2);
                          }
                        } catch {
                          reflectionOutputStr = iter.reflectionOutput || "";
                        }

                        return (
                          <div className="space-y-4">
                            {reflectionInputStr && (
                              <div className="border rounded-lg p-4 bg-muted/30">
                                <h4 className="font-medium text-sm mb-2 text-muted-foreground">
                                  Reflection Input (Dataset)
                                </h4>
                                <pre className="p-3 bg-muted rounded text-xs overflow-x-auto whitespace-pre-wrap max-h-64 overflow-y-auto">
                                  {reflectionInputStr}
                                </pre>
                              </div>
                            )}
                            {reflectionOutputStr && (
                              <div className="border rounded-lg p-4 bg-muted/30">
                                <h4 className="font-medium text-sm mb-2 text-muted-foreground">
                                  Proposed Instructions
                                </h4>
                                <pre className="p-3 bg-muted rounded text-xs overflow-x-auto whitespace-pre-wrap max-h-64 overflow-y-auto">
                                  {reflectionOutputStr}
                                </pre>
                              </div>
                            )}
                          </div>
                        );
                      }

                      // Fallback to LM calls (legacy dspy_gepa_logger)
                      if (reflectionCalls.length > 0) {
                        return reflectionCalls.map((call) => (
                          <div key={call.callId}>
                            {renderLmCallDetails(call)}
                          </div>
                        ));
                      }

                      return (
                        <p className="text-muted-foreground text-center py-8">
                          No reflection data recorded for this iteration.
                        </p>
                      );
                    })()}
                  </TabsContent>

                  <TabsContent value="evals" className="mt-4">
                    <PerformanceComparisonTable
                      baseEvaluations={parentEvals}
                      compareEvaluations={proposedEvals}
                      baseLabel={`Parent (#${iter.parentCandidateIdx ?? "?"})`}
                      compareLabel="Proposed"
                    />
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
