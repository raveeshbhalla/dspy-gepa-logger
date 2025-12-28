"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

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

type EvaluationModalProps = {
  entry: ComparisonEntry | null;
  entries: ComparisonEntry[];
  currentIndex: number;
  onNavigate: (index: number) => void;
  onClose: () => void;
};

type PredictionDisplayProps = {
  title: string;
  predictionRef: Record<string, unknown> | null;
  predictionPreview: string | null;
  variant: "seed" | "best";
};

/**
 * Parse a DSPy Prediction string representation like:
 * Prediction(reasoning="...", answer='4')
 *
 * Returns extracted fields or null if parsing fails.
 */
function parsePredictionString(predStr: string): Record<string, string> | null {
  // Match Prediction(...) or similar patterns
  const predMatch = predStr.match(/^(?:Prediction|ChainOfThought)\(([\s\S]*)\)$/);
  if (!predMatch) return null;

  const content = predMatch[1];
  const fields: Record<string, string> = {};

  // Parse key=value pairs, handling quoted strings
  // Match patterns like: reasoning="...", answer='4'
  const regex = /(\w+)=(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')/g;
  let match;
  while ((match = regex.exec(content)) !== null) {
    const key = match[1];
    const value = match[2] ?? match[3] ?? "";
    // Unescape common escapes
    fields[key] = value.replace(/\\\\/g, "\\").replace(/\\n/g, "\n").replace(/\\"/g, '"').replace(/\\'/g, "'");
  }

  return Object.keys(fields).length > 0 ? fields : null;
}

/**
 * Extract structured fields from a prediction reference object.
 * Handles various formats from DSPy predictions including nested completions.
 * Falls back to parsing the prediction preview string if ref is empty.
 */
function extractPredictionFields(
  predictionRef: Record<string, unknown> | null,
  predictionPreview: string | null = null
): Record<string, string> | null {
  // First try to extract from predictionRef object
  if (predictionRef && typeof predictionRef === "object") {
    const fields: Record<string, string> = {};

    for (const [key, value] of Object.entries(predictionRef)) {
      // Skip internal/metadata fields
      if (key.startsWith("_") || key === "completions") continue;

      if (value !== null && value !== undefined) {
        // Handle nested objects by stringifying them
        if (typeof value === "object") {
          fields[key] = JSON.stringify(value, null, 2);
        } else {
          fields[key] = String(value);
        }
      }
    }

    if (Object.keys(fields).length > 0) {
      return fields;
    }
  }

  // Fallback: try to parse the preview string (e.g., "Prediction(reasoning=..., answer='4')")
  if (predictionPreview) {
    return parsePredictionString(predictionPreview);
  }

  return null;
}

type PredictionDisplayPropsWithDelta = PredictionDisplayProps & {
  isRegression?: boolean;
};

/**
 * Display prediction fields from dspy.Predict or dspy.ChainOfThought.
 * Shows structured fields when predictionRef is available,
 * falls back to predictionPreview string otherwise.
 */
function PredictionDisplay({
  title,
  predictionRef,
  predictionPreview,
  variant,
  isRegression = false,
}: PredictionDisplayPropsWithDelta) {
  // For "best" variant, use red styling if it's a regression
  const containerClass = variant === "seed"
    ? "bg-muted/30 border border-border"
    : isRegression
      ? "bg-red-500/5 border border-red-500/20"
      : "bg-green-500/5 border border-green-500/20";

  // Extract structured fields from predictionRef, falling back to parsing the preview string
  const fields = extractPredictionFields(predictionRef, predictionPreview);

  // If we have structured prediction data, show each field
  if (fields && Object.keys(fields).length > 0) {
    // Get field names, prioritizing 'reasoning' first (for ChainOfThought),
    // then other fields alphabetically
    const fieldNames = Object.keys(fields);
    const sortedFields = fieldNames.sort((a, b) => {
      if (a === "reasoning") return -1;
      if (b === "reasoning") return 1;
      return a.localeCompare(b);
    });

    return (
      <div className="space-y-3">
        {sortedFields.map((fieldName) => {
          const value = fields[fieldName];
          const isReasoning = fieldName === "reasoning";
          return (
            <div key={fieldName} className={`${containerClass} p-3 rounded-lg`}>
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-1">
                {fieldName}
              </p>
              <p
                className={`text-sm whitespace-pre-wrap break-words ${
                  isReasoning ? "text-muted-foreground italic" : ""
                }`}
              >
                {value}
              </p>
            </div>
          );
        })}
      </div>
    );
  }

  // Fall back to preview string
  return (
    <div className={`${containerClass} p-3 rounded-lg`}>
      <p className="text-sm whitespace-pre-wrap break-words">
        {predictionPreview || "(no prediction captured)"}
      </p>
    </div>
  );
}

export function EvaluationModal({ entry, entries, currentIndex, onNavigate, onClose }: EvaluationModalProps) {
  if (!entry) return null;

  const canGoPrev = currentIndex > 0;
  const canGoNext = currentIndex < entries.length - 1;
  const isRegression = entry.delta < 0;

  // Dynamic styling for the Best column based on improvement/regression
  const bestBgClass = isRegression ? "bg-red-500/5 border border-red-500/20" : "bg-green-500/5 border border-green-500/20";
  const bestTextClass = isRegression ? "text-red-500" : "text-green-500";

  return (
    <Dialog open={!!entry} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <span>Evaluation Details</span>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => onNavigate(currentIndex - 1)}
                disabled={!canGoPrev}
                className="h-8 w-8"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-sm text-muted-foreground px-2">
                {currentIndex + 1} / {entries.length}
              </span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => onNavigate(currentIndex + 1)}
                disabled={!canGoNext}
                className="h-8 w-8"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {/* Input Fields - always show section */}
          <div>
            <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">Input</h4>
            {entry.inputs && Object.keys(entry.inputs).length > 0 ? (
              <div className="space-y-3">
                {Object.entries(entry.inputs).map(([key, value]) => (
                  <div key={key} className="bg-muted/50 p-3 rounded-lg">
                    <p className="text-xs font-medium text-muted-foreground mb-1">{key}</p>
                    <p className="text-sm whitespace-pre-wrap break-words">
                      {String(value)}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="bg-muted/50 p-3 rounded-lg">
                <p className="text-sm text-muted-foreground">(no input data captured)</p>
              </div>
            )}
          </div>

          {/* Side-by-side Score Comparison */}
          <div className="grid grid-cols-2 gap-4">
            {/* Seed Column */}
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Seed</h4>
              <div className="bg-muted/30 border border-border rounded-lg p-3">
                <p className="text-xs font-medium text-muted-foreground mb-1">Score</p>
                <p className="text-2xl font-semibold">{(entry.seedScore * 100).toFixed(0)}%</p>
              </div>
              <div className="bg-muted/30 border border-border rounded-lg p-3">
                <p className="text-xs font-medium text-muted-foreground mb-1">Feedback</p>
                <p className="text-sm whitespace-pre-wrap break-words">
                  {entry.seedFeedback || <span className="text-muted-foreground">(no feedback)</span>}
                </p>
              </div>
            </div>

            {/* Best Column */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Best</h4>
                <span
                  className={`text-xs px-1.5 py-0.5 rounded ${
                    entry.delta > 0
                      ? "bg-green-500/20 text-green-500"
                      : entry.delta < 0
                      ? "bg-red-500/20 text-red-500"
                      : "bg-muted text-muted-foreground"
                  }`}
                >
                  {entry.delta > 0 ? "+" : ""}
                  {(entry.delta * 100).toFixed(0)}%
                </span>
              </div>
              <div className={`${bestBgClass} rounded-lg p-3`}>
                <p className="text-xs font-medium text-muted-foreground mb-1">Score</p>
                <p className={`text-2xl font-semibold ${bestTextClass}`}>{(entry.bestScore * 100).toFixed(0)}%</p>
              </div>
              <div className={`${bestBgClass} rounded-lg p-3`}>
                <p className="text-xs font-medium text-muted-foreground mb-1">Feedback</p>
                <p className="text-sm whitespace-pre-wrap break-words">
                  {entry.bestFeedback || <span className="text-muted-foreground">(no feedback)</span>}
                </p>
              </div>
            </div>
          </div>

          {/* Output Fields - side by side */}
          <div>
            <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">Output</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-2">Seed</p>
                <PredictionDisplay
                  title="Seed"
                  predictionRef={entry.seedPredictionRef}
                  predictionPreview={entry.seedPrediction}
                  variant="seed"
                />
              </div>
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-2">Best</p>
                <PredictionDisplay
                  title="Best"
                  predictionRef={entry.bestPredictionRef}
                  predictionPreview={entry.bestPrediction}
                  variant="best"
                  isRegression={isRegression}
                />
              </div>
            </div>
          </div>

          {/* Example ID */}
          <p className="text-xs text-muted-foreground">
            Example ID: {entry.exampleId}
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
