"use client";

import { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import { computeLineDiff, hasChanges } from "./diff-utils";
import { UnifiedDiff } from "./UnifiedDiff";
import { SideBySideDiff } from "./SideBySideDiff";
import type { DiffViewerProps, DiffViewMode } from "./types";

export function DiffViewer({
  oldText,
  newText,
  oldLabel = "Original",
  newLabel = "Modified",
  defaultMode = "side-by-side",
  showLineNumbers = true,
  className,
}: DiffViewerProps) {
  const [mode, setMode] = useState<DiffViewMode>(defaultMode);

  const changed = hasChanges(oldText, newText);

  const diff = useMemo(
    () => (changed ? computeLineDiff(oldText, newText) : []),
    [oldText, newText, changed]
  );

  if (!changed) {
    return (
      <div className={cn("p-4 bg-muted/50 rounded-lg text-center", className)}>
        <p className="text-sm text-muted-foreground">No changes detected</p>
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      {/* Toggle */}
      <div className="flex justify-end gap-1">
        <button
          type="button"
          onClick={() => setMode("side-by-side")}
          className={cn(
            "px-3 py-1 text-xs rounded-md transition-colors",
            mode === "side-by-side"
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground hover:bg-muted/80"
          )}
        >
          Side by Side
        </button>
        <button
          type="button"
          onClick={() => setMode("unified")}
          className={cn(
            "px-3 py-1 text-xs rounded-md transition-colors",
            mode === "unified"
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground hover:bg-muted/80"
          )}
        >
          Unified
        </button>
      </div>

      {/* Diff view */}
      {mode === "unified" ? (
        <UnifiedDiff diff={diff} showLineNumbers={showLineNumbers} />
      ) : (
        <SideBySideDiff
          diff={diff}
          oldLabel={oldLabel}
          newLabel={newLabel}
          showLineNumbers={showLineNumbers}
        />
      )}
    </div>
  );
}
