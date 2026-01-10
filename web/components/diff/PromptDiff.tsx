"use client";

import { Badge } from "@/components/ui/badge";
import { DiffViewer } from "./DiffViewer";
import type { PromptDiffProps } from "./types";

export function PromptDiff({
  oldPrompt,
  newPrompt,
  oldLabel = "Original",
  newLabel = "Modified",
  defaultMode = "side-by-side",
}: PromptDiffProps) {
  const allKeys = [
    ...new Set([...Object.keys(oldPrompt), ...Object.keys(newPrompt)]),
  ];

  const hasAnyChanges = allKeys.some(
    (key) => (oldPrompt[key] || "") !== (newPrompt[key] || "")
  );

  if (!hasAnyChanges) {
    return (
      <div className="p-4 bg-muted/50 rounded-lg text-center">
        <p className="text-sm text-muted-foreground">
          No prompt changes detected
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {allKeys.map((key) => {
        const oldVal = oldPrompt[key] || "";
        const newVal = newPrompt[key] || "";
        const changed = oldVal !== newVal;

        return (
          <div key={key} className="space-y-2">
            <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
              {key}
              {changed && (
                <Badge variant="outline" className="text-yellow-500 border-yellow-500/50">
                  Changed
                </Badge>
              )}
            </h4>
            {changed ? (
              <DiffViewer
                oldText={oldVal}
                newText={newVal}
                oldLabel={oldLabel}
                newLabel={newLabel}
                defaultMode={defaultMode}
              />
            ) : (
              <div className="p-3 bg-muted rounded-lg">
                <pre className="whitespace-pre-wrap font-mono text-xs">
                  {oldVal || "(empty)"}
                </pre>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
