"use client";

import { cn } from "@/lib/utils";
import type { LineDiff, DiffSegment } from "./types";

type SideBySideDiffProps = {
  diff: LineDiff[];
  oldLabel?: string;
  newLabel?: string;
  showLineNumbers?: boolean;
  className?: string;
};

function renderSegments(
  segments: DiffSegment[] | undefined,
  highlightOperation: "delete" | "insert"
) {
  if (!segments) return null;

  return segments.map((seg, idx) => {
    const isHighlight = seg.operation === highlightOperation;
    return (
      <span
        key={idx}
        className={cn(
          isHighlight &&
            highlightOperation === "delete" &&
            "bg-red-500/40 rounded-sm",
          isHighlight &&
            highlightOperation === "insert" &&
            "bg-green-500/40 rounded-sm"
        )}
      >
        {seg.text}
      </span>
    );
  });
}

export function SideBySideDiff({
  diff,
  oldLabel = "Original",
  newLabel = "Modified",
  showLineNumbers = true,
  className,
}: SideBySideDiffProps) {
  // Pair up lines for side-by-side display
  // delete + insert pairs become single rows
  // equal lines become single rows
  // lone deletes/inserts get an empty opposite side

  type PairedLine = {
    left: LineDiff | null;
    right: LineDiff | null;
  };

  const pairedLines: PairedLine[] = [];
  let i = 0;

  while (i < diff.length) {
    const current = diff[i];

    if (current.operation === "equal") {
      pairedLines.push({ left: current, right: current });
      i++;
    } else if (current.operation === "delete") {
      // Check if next is an insert (modification pair)
      if (i + 1 < diff.length && diff[i + 1].operation === "insert") {
        pairedLines.push({ left: current, right: diff[i + 1] });
        i += 2;
      } else {
        pairedLines.push({ left: current, right: null });
        i++;
      }
    } else if (current.operation === "insert") {
      pairedLines.push({ left: null, right: current });
      i++;
    } else {
      i++;
    }
  }

  return (
    <div
      className={cn(
        "rounded-lg border overflow-hidden font-mono text-xs",
        className
      )}
    >
      {/* Header */}
      <div className="grid grid-cols-2 border-b bg-muted/50">
        <div className="px-3 py-2 font-medium text-muted-foreground border-r">
          {oldLabel}
        </div>
        <div className="px-3 py-2 font-medium text-muted-foreground">
          {newLabel}
        </div>
      </div>

      {/* Content */}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <tbody>
            {pairedLines.map((pair, idx) => {
              const leftIsDelete = pair.left?.operation === "delete";
              const rightIsInsert = pair.right?.operation === "insert";

              return (
                <tr key={idx}>
                  {/* Left side */}
                  <td
                    className={cn(
                      "border-r w-1/2 align-top",
                      leftIsDelete && "bg-red-500/10"
                    )}
                  >
                    <div className="flex">
                      {showLineNumbers && (
                        <span className="select-none text-right px-2 py-0.5 text-muted-foreground border-r border-border/50 w-10 shrink-0">
                          {pair.left?.lineNumber.old ?? ""}
                        </span>
                      )}
                      <div className="whitespace-pre-wrap break-all py-0.5 px-2 flex-1">
                        {pair.left ? (
                          pair.left.segments ? (
                            renderSegments(pair.left.segments, "delete")
                          ) : (
                            pair.left.content
                          )
                        ) : null}
                      </div>
                    </div>
                  </td>

                  {/* Right side */}
                  <td
                    className={cn(
                      "w-1/2 align-top",
                      rightIsInsert && "bg-green-500/10"
                    )}
                  >
                    <div className="flex">
                      {showLineNumbers && (
                        <span className="select-none text-right px-2 py-0.5 text-muted-foreground border-r border-border/50 w-10 shrink-0">
                          {pair.right?.lineNumber.new ?? ""}
                        </span>
                      )}
                      <div className="whitespace-pre-wrap break-all py-0.5 px-2 flex-1">
                        {pair.right ? (
                          pair.right.segments ? (
                            renderSegments(pair.right.segments, "insert")
                          ) : (
                            pair.right.content
                          )
                        ) : null}
                      </div>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
