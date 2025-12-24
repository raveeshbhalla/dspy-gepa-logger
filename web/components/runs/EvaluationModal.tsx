"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

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
  onClose: () => void;
};

type PredictionDisplayProps = {
  title: string;
  predictionRef: Record<string, unknown> | null;
  predictionPreview: string | null;
  variant: "seed" | "best";
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
}: PredictionDisplayProps) {
  const bgClass = variant === "seed" ? "bg-red-500/10" : "bg-green-500/10";

  // If we have structured prediction data, show each field
  if (predictionRef && typeof predictionRef === "object" && Object.keys(predictionRef).length > 0) {
    // Get field names, prioritizing 'reasoning' first (for ChainOfThought),
    // then other fields alphabetically
    const fields = Object.keys(predictionRef);
    const sortedFields = fields.sort((a, b) => {
      if (a === "reasoning") return -1;
      if (b === "reasoning") return 1;
      return a.localeCompare(b);
    });

    return (
      <div>
        <h4 className="text-sm font-medium mb-2">{title}</h4>
        <div className={`${bgClass} p-3 rounded-lg space-y-3`}>
          {sortedFields.map((fieldName) => {
            const value = predictionRef[fieldName];
            const isReasoning = fieldName === "reasoning";
            return (
              <div key={fieldName}>
                <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-1">
                  {fieldName}
                </p>
                <p
                  className={`text-sm whitespace-pre-wrap break-words ${
                    isReasoning ? "text-muted-foreground italic" : ""
                  }`}
                >
                  {String(value)}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // Fall back to preview string
  return (
    <div>
      <h4 className="text-sm font-medium mb-2">{title}</h4>
      <div className={`${bgClass} p-3 rounded-lg`}>
        <p className="text-sm whitespace-pre-wrap break-words">
          {predictionPreview || "(no prediction captured)"}
        </p>
      </div>
    </div>
  );
}

export function EvaluationModal({ entry, onClose }: EvaluationModalProps) {
  if (!entry) return null;

  return (
    <Dialog open={!!entry} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <span>Evaluation Details</span>
            <span
              className={`text-sm font-normal px-2 py-0.5 rounded ${
                entry.delta > 0
                  ? "bg-green-500/20 text-green-500"
                  : entry.delta < 0
                  ? "bg-red-500/20 text-red-500"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              {entry.delta > 0 ? "+" : ""}
              {(entry.delta * 100).toFixed(1)}%
            </span>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {/* Scores */}
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-3 bg-red-500/10 rounded-lg">
              <p className="text-2xl font-semibold">
                {(entry.seedScore * 100).toFixed(0)}%
              </p>
              <p className="text-xs text-muted-foreground">Seed Score</p>
            </div>
            <div className="text-center p-3 bg-green-500/10 rounded-lg">
              <p className="text-2xl font-semibold">
                {(entry.bestScore * 100).toFixed(0)}%
              </p>
              <p className="text-xs text-muted-foreground">Best Score</p>
            </div>
            <div
              className={`text-center p-3 rounded-lg ${
                entry.delta > 0
                  ? "bg-green-500/10"
                  : entry.delta < 0
                  ? "bg-red-500/10"
                  : "bg-muted"
              }`}
            >
              <p
                className={`text-2xl font-semibold ${
                  entry.delta > 0
                    ? "text-green-500"
                    : entry.delta < 0
                    ? "text-red-500"
                    : ""
                }`}
              >
                {entry.delta > 0 ? "+" : ""}
                {(entry.delta * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground">Delta</p>
            </div>
          </div>

          {/* Inputs */}
          {entry.inputs && Object.keys(entry.inputs).length > 0 && (
            <div>
              <h4 className="text-sm font-medium mb-2">Inputs</h4>
              <div className="bg-muted p-3 rounded-lg space-y-2">
                {Object.entries(entry.inputs).map(([key, value]) => (
                  <div key={key}>
                    <p className="text-xs text-muted-foreground">{key}</p>
                    <p className="text-sm whitespace-pre-wrap break-words">
                      {String(value)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Predictions */}
          <div className="grid grid-cols-2 gap-4">
            <PredictionDisplay
              title="Seed Prediction"
              predictionRef={entry.seedPredictionRef}
              predictionPreview={entry.seedPrediction}
              variant="seed"
            />
            <PredictionDisplay
              title="Best Prediction"
              predictionRef={entry.bestPredictionRef}
              predictionPreview={entry.bestPrediction}
              variant="best"
            />
          </div>

          {/* Feedback */}
          {(entry.seedFeedback || entry.bestFeedback) && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-medium mb-2">Seed Feedback</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="text-sm whitespace-pre-wrap break-words">
                    {entry.seedFeedback || "(no feedback)"}
                  </p>
                </div>
              </div>
              <div>
                <h4 className="text-sm font-medium mb-2">Best Feedback</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="text-sm whitespace-pre-wrap break-words">
                    {entry.bestFeedback || "(no feedback)"}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Example ID */}
          <p className="text-xs text-muted-foreground">
            Example ID: {entry.exampleId}
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
