"use client";

import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

type LogEntry = {
  id: string;
  logType: string;
  content: string;
  timestamp: number;
  iteration?: number | null;
  phase?: string | null;
};

type LmCallContent = {
  call_id: string;
  model: string;
  duration_ms: number;
  iteration?: number | null;
  phase?: string | null;
  candidate_idx?: number | null;
  inputs_preview?: string | null;
  outputs_preview?: string | null;
};

type LogsTabProps = {
  runId: string;
  isRunning: boolean;
};

export function LogsTab({ runId, isRunning }: LogsTabProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Fetch initial logs
  useEffect(() => {
    // Clear logs immediately when runId changes to prevent mixing old/new logs
    setLogs([]);

    async function fetchLogs() {
      try {
        const res = await fetch(`/api/runs/${runId}/logs`);
        if (res.ok) {
          const data = await res.json();
          setLogs(data);
        }
      } catch (error) {
        console.error("Error fetching logs:", error);
      }
    }
    fetchLogs();
  }, [runId]);

  // Subscribe to SSE for real-time updates
  useEffect(() => {
    if (!isRunning) return;

    const eventSource = new EventSource(`/api/events/${runId}`);
    eventSourceRef.current = eventSource;

    eventSource.addEventListener("log", (event) => {
      try {
        const logData = JSON.parse(event.data);
        setLogs((prev) => [...prev, logData]);
      } catch (error) {
        console.error("Error parsing log event:", error);
      }
    });

    eventSource.onerror = () => {
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [runId, isRunning]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  function formatTimestamp(timestamp: number): string {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      fractionalSecondDigits: 3,
    });
  }

  function getLogTypeColor(logType: string): string {
    switch (logType) {
      case "stdout":
        return "text-foreground";
      case "stderr":
        return "text-destructive";
      case "lm_call":
        return "text-blue-400";
      case "info":
        return "text-muted-foreground";
      default:
        return "text-foreground";
    }
  }

  function getLogTypeBadgeVariant(logType: string): "default" | "secondary" | "destructive" | "outline" {
    switch (logType) {
      case "stderr":
        return "destructive";
      case "lm_call":
        return "default";
      default:
        return "secondary";
    }
  }

  function renderLogContent(log: LogEntry) {
    if (log.logType === "lm_call") {
      try {
        const lmCall: LmCallContent = JSON.parse(log.content);
        return (
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="font-medium text-blue-400">LM Call</span>
              <Badge variant="outline" className="text-xs">
                {lmCall.model}
              </Badge>
              <span className="text-xs text-muted-foreground">
                {lmCall.duration_ms?.toFixed(0)}ms
              </span>
              {lmCall.phase && (
                <Badge variant="secondary" className="text-xs">
                  {lmCall.phase}
                </Badge>
              )}
            </div>
            {lmCall.inputs_preview && (
              <details className="text-xs">
                <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                  Inputs
                </summary>
                <pre className="mt-1 p-2 bg-muted/50 rounded text-xs overflow-x-auto whitespace-pre-wrap">
                  {lmCall.inputs_preview}
                </pre>
              </details>
            )}
            {lmCall.outputs_preview && (
              <details className="text-xs">
                <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                  Outputs
                </summary>
                <pre className="mt-1 p-2 bg-muted/50 rounded text-xs overflow-x-auto whitespace-pre-wrap">
                  {lmCall.outputs_preview}
                </pre>
              </details>
            )}
          </div>
        );
      } catch {
        return <span className="whitespace-pre-wrap">{log.content}</span>;
      }
    }

    return <span className="whitespace-pre-wrap">{log.content}</span>;
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Logs</CardTitle>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-muted-foreground">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="rounded"
              />
              Auto-scroll
            </label>
            {isRunning && (
              <Badge variant="default" className="animate-pulse">
                Live
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[600px] rounded-md border">
          <div
            ref={scrollRef}
            className="p-4 font-mono text-sm space-y-1"
          >
            {logs.length === 0 ? (
              <p className="text-muted-foreground text-center py-8">
                No logs yet. {isRunning ? "Waiting for output..." : "Run has no logs."}
              </p>
            ) : (
              logs.map((log) => (
                <div
                  key={log.id}
                  className={`flex gap-3 ${getLogTypeColor(log.logType)}`}
                >
                  <span className="text-muted-foreground shrink-0">
                    {formatTimestamp(log.timestamp)}
                  </span>
                  <Badge
                    variant={getLogTypeBadgeVariant(log.logType)}
                    className="shrink-0 text-xs h-5"
                  >
                    {log.logType}
                  </Badge>
                  {log.iteration != null && (
                    <span className="text-muted-foreground shrink-0 text-xs">
                      [iter {log.iteration + 1}]
                    </span>
                  )}
                  <div className="flex-1 min-w-0">
                    {renderLogContent(log)}
                  </div>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
