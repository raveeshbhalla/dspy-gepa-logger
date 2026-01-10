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
 * Calculate similarity ratio between two strings (0 to 1)
 * Uses Levenshtein distance normalized by max length
 */
function calculateSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  if (a.length === 0 || b.length === 0) return 0;

  // Use diff-match-patch to compute Levenshtein distance
  const diffs = dmp.diff_main(a, b);
  const levenshtein = dmp.diff_levenshtein(diffs);
  const maxLen = Math.max(a.length, b.length);

  return 1 - levenshtein / maxLen;
}

/**
 * Match deleted lines with inserted lines using similarity.
 * Returns pairs of indices and unmatched lines.
 */
function matchLines(
  deletedLines: LineDiff[],
  insertedLines: LineDiff[]
): {
  pairs: Array<{ deleteIdx: number; insertIdx: number; similarity: number }>;
  unmatchedDeletes: number[];
  unmatchedInserts: number[];
} {
  const SIMILARITY_THRESHOLD = 0.4; // Minimum similarity to consider a match

  // Build similarity matrix
  const similarities: Array<{
    deleteIdx: number;
    insertIdx: number;
    similarity: number;
  }> = [];

  for (let d = 0; d < deletedLines.length; d++) {
    for (let i = 0; i < insertedLines.length; i++) {
      const sim = calculateSimilarity(
        deletedLines[d].content,
        insertedLines[i].content
      );
      if (sim >= SIMILARITY_THRESHOLD) {
        similarities.push({ deleteIdx: d, insertIdx: i, similarity: sim });
      }
    }
  }

  // Sort by similarity (highest first)
  similarities.sort((a, b) => b.similarity - a.similarity);

  // Greedy matching - take best matches first
  const usedDeletes = new Set<number>();
  const usedInserts = new Set<number>();
  const pairs: Array<{
    deleteIdx: number;
    insertIdx: number;
    similarity: number;
  }> = [];

  for (const match of similarities) {
    if (
      !usedDeletes.has(match.deleteIdx) &&
      !usedInserts.has(match.insertIdx)
    ) {
      pairs.push(match);
      usedDeletes.add(match.deleteIdx);
      usedInserts.add(match.insertIdx);
    }
  }

  // Find unmatched lines
  const unmatchedDeletes: number[] = [];
  const unmatchedInserts: number[] = [];

  for (let d = 0; d < deletedLines.length; d++) {
    if (!usedDeletes.has(d)) unmatchedDeletes.push(d);
  }
  for (let i = 0; i < insertedLines.length; i++) {
    if (!usedInserts.has(i)) unmatchedInserts.push(i);
  }

  return { pairs, unmatchedDeletes, unmatchedInserts };
}

/**
 * Add character-level segments to a delete/insert pair
 */
function enhancePair(
  deleteLine: LineDiff,
  insertLine: LineDiff
): { enhanced: LineDiff; enhancedInsert: LineDiff } {
  const charDiff = computeCharacterDiff(deleteLine.content, insertLine.content);

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

  return {
    enhanced: { ...deleteLine, segments: deleteSegments },
    enhancedInsert: { ...insertLine, segments: insertSegments },
  };
}

/**
 * Compute line-by-line diff with inline character-level highlighting.
 * Uses diff-match-patch's line mode for accurate line-level diffing,
 * then uses similarity matching to pair modified lines for character-level diffs.
 */
export function computeLineDiff(oldText: string, newText: string): LineDiff[] {
  // Use diff-match-patch's line mode diffing
  const { chars1, chars2, lineArray } = dmp.diff_linesToChars_(
    oldText,
    newText
  );
  const diffs = dmp.diff_main(chars1, chars2, false);
  dmp.diff_charsToLines_(diffs, lineArray);
  dmp.diff_cleanupSemantic(diffs);

  // Convert to our LineDiff format
  const result: LineDiff[] = [];
  let oldLineNum = 1;
  let newLineNum = 1;

  for (const [op, text] of diffs) {
    const operation = toOperation(op);
    const lines = text.split("\n");
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

  // Process change blocks and match similar lines
  const enhancedResult: LineDiff[] = [];
  let i = 0;

  while (i < result.length) {
    const current = result[i];

    if (current.operation === "equal") {
      enhancedResult.push(current);
      i++;
      continue;
    }

    // Collect contiguous block of deletes and inserts
    const deleteLines: LineDiff[] = [];
    const insertLines: LineDiff[] = [];
    let j = i;

    // First collect all deletes
    while (j < result.length && result[j].operation === "delete") {
      deleteLines.push(result[j]);
      j++;
    }

    // Then collect all inserts
    while (j < result.length && result[j].operation === "insert") {
      insertLines.push(result[j]);
      j++;
    }

    // If we have both deletes and inserts, try to match them
    if (deleteLines.length > 0 && insertLines.length > 0) {
      const { pairs, unmatchedDeletes, unmatchedInserts } = matchLines(
        deleteLines,
        insertLines
      );

      // Sort pairs by delete index to maintain order
      pairs.sort((a, b) => a.deleteIdx - b.deleteIdx);

      // Build output: interleave matched pairs, put unmatched at appropriate positions
      let deletePtr = 0;
      let insertPtr = 0;
      let pairPtr = 0;

      // Track which inserts have been used
      const usedInserts = new Set(pairs.map((p) => p.insertIdx));

      while (deletePtr < deleteLines.length) {
        // Check if current delete is part of a pair
        const pairForThisDelete = pairs.find(
          (p) => p.deleteIdx === deletePtr
        );

        if (pairForThisDelete) {
          // Output the matched pair with character-level diff
          const { enhanced, enhancedInsert } = enhancePair(
            deleteLines[deletePtr],
            insertLines[pairForThisDelete.insertIdx]
          );
          enhancedResult.push(enhanced);
          enhancedResult.push(enhancedInsert);
        } else {
          // Unmatched delete
          enhancedResult.push(deleteLines[deletePtr]);
        }
        deletePtr++;
      }

      // Add any remaining unmatched inserts
      for (const insertIdx of unmatchedInserts) {
        enhancedResult.push(insertLines[insertIdx]);
      }
    } else {
      // Only deletes or only inserts - just add them
      for (const line of deleteLines) {
        enhancedResult.push(line);
      }
      for (const line of insertLines) {
        enhancedResult.push(line);
      }
    }

    i = j;
  }

  return enhancedResult;
}

/**
 * Check if two texts have any differences
 */
export function hasChanges(oldText: string, newText: string): boolean {
  return oldText !== newText;
}
