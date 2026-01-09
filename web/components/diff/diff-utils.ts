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
 * Uses diff-match-patch's line mode for accurate line-level diffing,
 * then enhances modified lines with character-level segments.
 */
export function computeLineDiff(oldText: string, newText: string): LineDiff[] {
  // Use diff-match-patch's line mode diffing
  // This properly handles line-to-character encoding/decoding
  const { chars1, chars2, lineArray } = dmp.diff_linesToChars_(oldText, newText);
  const diffs = dmp.diff_main(chars1, chars2, false);
  dmp.diff_charsToLines_(diffs, lineArray);
  dmp.diff_cleanupSemantic(diffs);

  // Convert to our LineDiff format
  const result: LineDiff[] = [];
  let oldLineNum = 1;
  let newLineNum = 1;

  for (const [op, text] of diffs) {
    const operation = toOperation(op);
    // Split text into lines, handling the trailing newline
    const lines = text.split("\n");
    // Remove empty string at end if text ended with \n
    if (lines.length > 0 && lines[lines.length - 1] === "") {
      lines.pop();
    }

    for (const lineContent of lines) {
      if (operation === "equal") {
        result.push({
          lineNumber: { old: oldLineNum, new: newLineNum },
          operation: "equal",
          content: lineContent,
        });
        oldLineNum++;
        newLineNum++;
      } else if (operation === "delete") {
        result.push({
          lineNumber: { old: oldLineNum, new: null },
          operation: "delete",
          content: lineContent,
        });
        oldLineNum++;
      } else if (operation === "insert") {
        result.push({
          lineNumber: { old: null, new: newLineNum },
          operation: "insert",
          content: lineContent,
        });
        newLineNum++;
      }
    }
  }

  // Enhance adjacent delete+insert pairs with character-level diffs
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
