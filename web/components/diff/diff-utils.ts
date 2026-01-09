import DiffMatchPatch from "diff-match-patch";
import type { DiffOperation, DiffSegment, LineDiff } from "./types";

const dmp = new DiffMatchPatch();

/**
 * Convert diff-match-patch operation to our DiffOperation type
 */
function toOperation(op: number): DiffOperation {
  if (op === DiffMatchPatch.DIFF_DELETE) return "delete";
  if (op === DiffMatchPatch.DIFF_INSERT) return "insert";
  return "equal";
}

/**
 * Compute character-level diff between two strings
 */
export function computeCharacterDiff(
  oldText: string,
  newText: string
): DiffSegment[] {
  const diffs = dmp.diff_main(oldText, newText);
  dmp.diff_cleanupSemantic(diffs);

  return diffs.map(([op, text]) => ({
    operation: toOperation(op),
    text,
  }));
}

/**
 * Compute line-by-line diff with inline character-level highlighting.
 * Uses Myers' diff algorithm via diff-match-patch for character-level diff,
 * then groups results by line. No unique line count limitations.
 */
export function computeLineDiff(oldText: string, newText: string): LineDiff[] {
  // Compute character-level diff
  const charDiffs = dmp.diff_main(oldText, newText);
  dmp.diff_cleanupSemantic(charDiffs);

  // Process diffs and group by lines
  const result: LineDiff[] = [];
  let oldLineNum = 1;
  let newLineNum = 1;

  // Current line buffers
  let oldLineBuffer = "";
  let newLineBuffer = "";
  let oldSegments: DiffSegment[] = [];
  let newSegments: DiffSegment[] = [];

  // Track if we're in a modification (delete+insert on same logical line)
  let pendingDelete: { content: string; segments: DiffSegment[] } | null = null;

  for (const [op, text] of charDiffs) {
    const operation = toOperation(op);
    const lines = text.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const lineContent = lines[i];
      const isLastPart = i === lines.length - 1;

      if (operation === "equal") {
        // Flush any pending modifications
        if (pendingDelete) {
          result.push({
            lineNumber: { old: oldLineNum, new: null },
            operation: "delete",
            content: pendingDelete.content,
            segments: pendingDelete.segments.length > 0 ? pendingDelete.segments : undefined,
          });
          oldLineNum++;
          pendingDelete = null;
        }

        oldLineBuffer += lineContent;
        newLineBuffer += lineContent;

        if (!isLastPart) {
          // Hit a newline - flush the equal line
          result.push({
            lineNumber: { old: oldLineNum, new: newLineNum },
            operation: "equal",
            content: oldLineBuffer,
          });
          oldLineNum++;
          newLineNum++;
          oldLineBuffer = "";
          newLineBuffer = "";
        }
      } else if (operation === "delete") {
        oldLineBuffer += lineContent;
        oldSegments.push({ operation: "delete", text: lineContent });

        if (!isLastPart) {
          // Store as pending - might be paired with insert
          pendingDelete = { content: oldLineBuffer, segments: [...oldSegments] };
          oldLineBuffer = "";
          oldSegments = [];
        }
      } else if (operation === "insert") {
        newLineBuffer += lineContent;
        newSegments.push({ operation: "insert", text: lineContent });

        if (!isLastPart) {
          if (pendingDelete) {
            // Pair delete+insert as a modification
            result.push({
              lineNumber: { old: oldLineNum, new: null },
              operation: "delete",
              content: pendingDelete.content,
              segments: pendingDelete.segments,
            });
            oldLineNum++;
            result.push({
              lineNumber: { old: null, new: newLineNum },
              operation: "insert",
              content: newLineBuffer,
              segments: [...newSegments],
            });
            newLineNum++;
            pendingDelete = null;
          } else {
            result.push({
              lineNumber: { old: null, new: newLineNum },
              operation: "insert",
              content: newLineBuffer,
              segments: newSegments.length > 0 ? [...newSegments] : undefined,
            });
            newLineNum++;
          }
          newLineBuffer = "";
          newSegments = [];
        }
      }
    }
  }

  // Flush remaining content
  if (pendingDelete) {
    result.push({
      lineNumber: { old: oldLineNum, new: null },
      operation: "delete",
      content: pendingDelete.content,
      segments: pendingDelete.segments.length > 0 ? pendingDelete.segments : undefined,
    });
    oldLineNum++;
  }
  if (oldLineBuffer || newLineBuffer) {
    if (oldLineBuffer === newLineBuffer && oldLineBuffer !== "") {
      result.push({
        lineNumber: { old: oldLineNum, new: newLineNum },
        operation: "equal",
        content: oldLineBuffer,
      });
    } else {
      if (oldLineBuffer) {
        result.push({
          lineNumber: { old: oldLineNum, new: null },
          operation: "delete",
          content: oldLineBuffer,
          segments: oldSegments.length > 0 ? oldSegments : undefined,
        });
      }
      if (newLineBuffer) {
        result.push({
          lineNumber: { old: null, new: newLineNum },
          operation: "insert",
          content: newLineBuffer,
          segments: newSegments.length > 0 ? newSegments : undefined,
        });
      }
    }
  }

  // Add character-level segments for modified lines (delete followed by insert)
  const enhancedResult: LineDiff[] = [];
  let i = 0;

  while (i < result.length) {
    const current = result[i];

    // Check if this is a delete followed by an insert (likely a modification)
    if (
      current.operation === "delete" &&
      i + 1 < result.length &&
      result[i + 1].operation === "insert"
    ) {
      const deleteItem = current;
      const insertItem = result[i + 1];

      // Compute character-level diff for the modification
      const charDiff = computeCharacterDiff(
        deleteItem.content,
        insertItem.content
      );

      // Split into delete and insert segments
      const deleteSegments: DiffSegment[] = [];
      const insertSegments: DiffSegment[] = [];

      for (const seg of charDiff) {
        if (seg.operation === "delete") {
          deleteSegments.push({ operation: "delete", text: seg.text });
        } else if (seg.operation === "insert") {
          insertSegments.push({ operation: "insert", text: seg.text });
        } else {
          deleteSegments.push({ operation: "equal", text: seg.text });
          insertSegments.push({ operation: "equal", text: seg.text });
        }
      }

      enhancedResult.push({
        ...deleteItem,
        segments: deleteSegments,
      });
      enhancedResult.push({
        ...insertItem,
        segments: insertSegments,
      });

      i += 2;
    } else {
      enhancedResult.push(current);
      i++;
    }
  }

  return enhancedResult;
}

/**
 * Check if two texts have any differences
 */
export function hasChanges(oldText: string, newText: string): boolean {
  return oldText !== newText;
}
