import * as Diff from "diff";
import type { DiffOperation, DiffSegment, LineDiff } from "./types";

/**
 * Compute character-level diff between two strings
 */
export function computeCharacterDiff(
  oldText: string,
  newText: string
): DiffSegment[] {
  const changes = Diff.diffChars(oldText, newText);

  return changes.map((change) => ({
    operation: change.added ? "insert" : change.removed ? "delete" : "equal",
    text: change.value,
  }));
}

/**
 * Compute line-by-line diff with inline character-level highlighting.
 * Uses jsdiff for line-level diffing.
 */
export function computeLineDiff(oldText: string, newText: string): LineDiff[] {
  const changes = Diff.diffLines(oldText, newText);

  const result: LineDiff[] = [];
  let oldLineNum = 1;
  let newLineNum = 1;

  for (const change of changes) {
    // Split change value into individual lines
    const lines = change.value.split("\n");
    // Remove trailing empty string from split (if value ended with \n)
    if (lines.length > 0 && lines[lines.length - 1] === "") {
      lines.pop();
    }

    const operation: DiffOperation = change.added
      ? "insert"
      : change.removed
        ? "delete"
        : "equal";

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

    // Check if this is a delete followed by an insert (modification)
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
