"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

type PromptComparisonProps = {
  seedPrompt: Record<string, string>;
  bestPrompt: Record<string, string>;
  seedCandidateIdx: number;
  bestCandidateIdx: number | null;
};

export function PromptComparison({
  seedPrompt,
  bestPrompt,
  seedCandidateIdx,
  bestCandidateIdx,
}: PromptComparisonProps) {
  const allKeys = [
    ...new Set([...Object.keys(seedPrompt), ...Object.keys(bestPrompt)]),
  ];

  const hasChanges = allKeys.some(
    (key) => seedPrompt[key] !== bestPrompt[key]
  );

  if (!hasChanges) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Prompt Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No prompt changes detected between seed and best candidate.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Prompt Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="side-by-side">
          <TabsList className="mb-4">
            <TabsTrigger value="side-by-side">Side by Side</TabsTrigger>
            <TabsTrigger value="original">Original (#{seedCandidateIdx})</TabsTrigger>
            <TabsTrigger value="optimized">
              Optimized{bestCandidateIdx != null ? ` (#${bestCandidateIdx})` : ""}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="side-by-side">
            <div className="space-y-4">
              {allKeys.map((key) => {
                const seedVal = seedPrompt[key] || "";
                const bestVal = bestPrompt[key] || "";
                const changed = seedVal !== bestVal;

                return (
                  <div key={key} className="space-y-2">
                    <h4 className="text-sm font-medium text-foreground">
                      {key}
                      {changed && (
                        <span className="ml-2 text-xs text-yellow-500">Changed</span>
                      )}
                    </h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div
                        className={`p-3 rounded-lg text-sm ${
                          changed
                            ? "bg-red-500/10 border border-red-500/20"
                            : "bg-muted"
                        }`}
                      >
                        <p className="text-xs text-muted-foreground mb-1">
                          Original
                        </p>
                        <pre className="whitespace-pre-wrap font-mono text-xs">
                          {seedVal || "(empty)"}
                        </pre>
                      </div>
                      <div
                        className={`p-3 rounded-lg text-sm ${
                          changed
                            ? "bg-green-500/10 border border-green-500/20"
                            : "bg-muted"
                        }`}
                      >
                        <p className="text-xs text-muted-foreground mb-1">
                          Optimized
                        </p>
                        <pre className="whitespace-pre-wrap font-mono text-xs">
                          {bestVal || "(empty)"}
                        </pre>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </TabsContent>

          <TabsContent value="original">
            <div className="space-y-4">
              {allKeys.map((key) => (
                <div key={key} className="space-y-2">
                  <h4 className="text-sm font-medium text-foreground">{key}</h4>
                  <div className="p-3 bg-muted rounded-lg">
                    <pre className="whitespace-pre-wrap font-mono text-xs">
                      {seedPrompt[key] || "(empty)"}
                    </pre>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="optimized">
            <div className="space-y-4">
              {allKeys.map((key) => (
                <div key={key} className="space-y-2">
                  <h4 className="text-sm font-medium text-foreground">{key}</h4>
                  <div className="p-3 bg-muted rounded-lg">
                    <pre className="whitespace-pre-wrap font-mono text-xs">
                      {bestPrompt[key] || "(empty)"}
                    </pre>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
