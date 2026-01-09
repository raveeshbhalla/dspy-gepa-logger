"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PromptDiff } from "@/components/diff";

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
        <Tabs defaultValue="diff">
          <TabsList className="mb-4">
            <TabsTrigger value="diff">Diff View</TabsTrigger>
            <TabsTrigger value="original">Original (#{seedCandidateIdx})</TabsTrigger>
            <TabsTrigger value="optimized">
              Optimized{bestCandidateIdx != null ? ` (#${bestCandidateIdx})` : ""}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="diff">
            <PromptDiff
              oldPrompt={seedPrompt}
              newPrompt={bestPrompt}
              oldLabel={`Original (#${seedCandidateIdx})`}
              newLabel={`Optimized${bestCandidateIdx != null ? ` (#${bestCandidateIdx})` : ""}`}
            />
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
