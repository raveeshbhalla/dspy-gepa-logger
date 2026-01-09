export type DiffOperation = "equal" | "insert" | "delete";

export type DiffSegment = {
  operation: DiffOperation;
  text: string;
};

export type LineDiff = {
  lineNumber: { old: number | null; new: number | null };
  operation: DiffOperation;
  content: string;
  segments?: DiffSegment[];
};

export type DiffViewMode = "unified" | "side-by-side";

export type DiffViewerProps = {
  oldText: string;
  newText: string;
  oldLabel?: string;
  newLabel?: string;
  defaultMode?: DiffViewMode;
  showLineNumbers?: boolean;
  className?: string;
};

export type PromptDiffProps = {
  oldPrompt: Record<string, string>;
  newPrompt: Record<string, string>;
  oldLabel?: string;
  newLabel?: string;
  defaultMode?: DiffViewMode;
};
