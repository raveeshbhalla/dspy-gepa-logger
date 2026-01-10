"use client";

import { cn } from "@/lib/utils";
import type { LineDiff, DiffSegment } from "./types";

type UnifiedDiffProps = {
  diff: LineDiff[];
  showLineNumbers?: boolean;
  className?: string;
};

function renderSegments(segments: DiffSegment[] | undefined, operation: "delete" | "insert") {
  if (!segments) return null;

  return segments.map((seg, idx) => {
    const isHighlight = seg.operation === operation;
    return (
      <span
        key={idx}
        className={cn(
          isHighlight && operation === "delete" && "bg-red-500/40 rounded-sm",
          isHighlight && operation === "insert" && "bg-green-500/40 rounded-sm"
        )}
      >
        {seg.text}
      </span>
    );
  });
}

export function UnifiedDiff({
  diff,
  showLineNumbers = true,
  className,
}: UnifiedDiffProps) {
  return (
    <div
      className={cn(
        "rounded-lg border overflow-hidden font-mono text-xs",
        className
      )}
    >
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <tbody>
            {diff.map((line, idx) => {
              const isDelete = line.operation === "delete";
              const isInsert = line.operation === "insert";
              const prefix = isDelete ? "-" : isInsert ? "+" : " ";

              return (
                <tr
                  key={idx}
                  className={cn(
                    isDelete && "bg-red-500/10",
                    isInsert && "bg-green-500/10"
                  )}
                >
                  {showLineNumbers && (
                    <>
                      <td className="select-none text-right px-2 py-0.5 text-muted-foreground border-r border-border/50 w-10 align-top">
                        {line.lineNumber.old ?? ""}
                      </td>
                      <td className="select-none text-right px-2 py-0.5 text-muted-foreground border-r border-border/50 w-10 align-top">
                        {line.lineNumber.new ?? ""}
                      </td>
                    </>
                  )}
                  <td
                    className={cn(
                      "select-none w-6 text-center align-top",
                      isDelete && "text-red-500",
                      isInsert && "text-green-500"
                    )}
                  >
                    {prefix}
                  </td>
                  <td className="whitespace-pre-wrap break-all py-0.5 pr-3 align-top">
                    {line.segments && line.segments.length > 0 && (isDelete || isInsert) ? (
                      renderSegments(line.segments, line.operation as "delete" | "insert")
                    ) : (
                      line.content
                    )}
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
