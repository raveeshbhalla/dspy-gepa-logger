"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useRouter, usePathname } from "next/navigation";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Trash2 } from "lucide-react";

type Project = {
  id: string;
  name: string;
  _count: { runs: number };
};

type Run = {
  id: string;
  name: string | null;
  status: "RUNNING" | "COMPLETED" | "FAILED";
  startedAt: string;
  totalIterations: number;
  bestScore: number | null;
};

export function Sidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const eventSourceRef = useRef<EventSource | null>(null);

  const fetchRuns = useCallback(async (projectId: string, setLoadingState = false) => {
    if (setLoadingState) setLoading(true);
    try {
      const res = await fetch(`/api/runs?projectId=${projectId}`);
      const data = await res.json();
      setRuns(data);
    } catch (error) {
      console.error("Failed to fetch runs:", error);
    } finally {
      if (setLoadingState) setLoading(false);
    }
  }, []);

  // Fetch projects on mount
  useEffect(() => {
    async function fetchProjects() {
      try {
        const res = await fetch("/api/projects");
        const data = await res.json();
        setProjects(data);
        if (data.length > 0 && !selectedProjectId) {
          setSelectedProjectId(data[0].id);
        }
      } catch (error) {
        console.error("Failed to fetch projects:", error);
      }
    }
    fetchProjects();
  }, []);

  // Fetch runs when project changes
  useEffect(() => {
    if (!selectedProjectId) return;
    fetchRuns(selectedProjectId, true);
  }, [selectedProjectId, fetchRuns]);

  // SSE for real-time updates (new runs, status changes)
  useEffect(() => {
    // Clean up previous connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const eventSource = new EventSource("/api/events");
    eventSourceRef.current = eventSource;

    eventSource.addEventListener("run_created", () => {
      // Refetch runs when a new run is created
      if (selectedProjectId) {
        fetchRuns(selectedProjectId);
      }
      // Also refetch projects to update run counts
      fetch("/api/projects")
        .then((res) => res.json())
        .then(setProjects)
        .catch(console.error);
    });

    eventSource.addEventListener("run_completed", () => {
      // Refetch runs when a run completes
      if (selectedProjectId) {
        fetchRuns(selectedProjectId);
      }
    });

    eventSource.onerror = () => {
      // Reconnect on error (will happen automatically, but we can log it)
      console.log("SSE connection lost, reconnecting...");
    };

    return () => {
      eventSource.close();
      eventSourceRef.current = null;
    };
  }, [selectedProjectId, fetchRuns]);

  const currentRunId = pathname?.startsWith("/runs/")
    ? pathname.split("/")[2]
    : null;

  const handleDeleteRun = useCallback(async (e: React.MouseEvent, runId: string) => {
    e.stopPropagation();

    try {
      const res = await fetch(`/api/runs/${runId}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to delete run");

      // Refresh runs list
      if (selectedProjectId) {
        fetchRuns(selectedProjectId);
      }

      // Update project counts
      fetch("/api/projects")
        .then((res) => res.json())
        .then(setProjects)
        .catch(console.error);

      // Navigate away if we deleted the current run
      if (currentRunId === runId) {
        router.push("/");
      }
    } catch (error) {
      console.error("Failed to delete run:", error);
    }
  }, [selectedProjectId, fetchRuns, currentRunId, router]);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="w-64 border-r border-border bg-card flex flex-col h-screen">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <h1 className="text-lg font-semibold text-foreground">GEPA Logger</h1>
      </div>

      {/* Project Selector */}
      <div className="p-4 border-b border-border">
        <label className="text-xs text-muted-foreground uppercase tracking-wide mb-2 block">
          Project
        </label>
        <Select value={selectedProjectId} onValueChange={setSelectedProjectId}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select project" />
          </SelectTrigger>
          <SelectContent>
            {projects.map((project) => (
              <SelectItem key={project.id} value={project.id}>
                {project.name} ({project._count.runs})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Run List */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="px-4 py-2">
          <span className="text-xs text-muted-foreground uppercase tracking-wide">
            Recent Runs
          </span>
        </div>
        <ScrollArea className="flex-1">
          <div className="px-2 pb-4">
            {loading ? (
              <div className="p-4 text-center text-muted-foreground text-sm">
                Loading...
              </div>
            ) : runs.length === 0 ? (
              <div className="p-4 text-center text-muted-foreground text-sm">
                No runs yet
              </div>
            ) : (
              runs.map((run) => (
                <button
                  key={run.id}
                  onClick={() => router.push(`/runs/${run.id}`)}
                  className={cn(
                    "w-full text-left p-3 rounded-lg mb-1 transition-colors group",
                    "hover:bg-accent",
                    currentRunId === run.id
                      ? "bg-accent"
                      : "bg-transparent"
                  )}
                >
                  <div className="flex items-center justify-between mb-1 gap-2">
                    <span className="text-sm font-medium truncate flex-1">
                      {run.name || `Run ${run.id.slice(0, 8)}`}
                    </span>
                    <button
                      onClick={(e) => handleDeleteRun(e, run.id)}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-destructive/20 hover:text-destructive transition-opacity"
                      title="Delete run"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                    <StatusBadge status={run.status} />
                  </div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{formatDate(run.startedAt)}</span>
                    <span>
                      {run.totalIterations} iter
                      {run.bestScore !== null &&
                        ` · ${(run.bestScore * 100).toFixed(0)}%`}
                    </span>
                  </div>
                </button>
              ))
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: "RUNNING" | "COMPLETED" | "FAILED" }) {
  const variants: Record<typeof status, "default" | "secondary" | "destructive" | "outline"> = {
    RUNNING: "default",
    COMPLETED: "secondary",
    FAILED: "destructive",
  };

  const labels: Record<typeof status, string> = {
    RUNNING: "Running",
    COMPLETED: "Done",
    FAILED: "Failed",
  };

  return (
    <Badge variant={variants[status]} className="text-[10px] px-1.5 py-0">
      {status === "RUNNING" && (
        <span className="w-1.5 h-1.5 rounded-full bg-current mr-1 animate-pulse" />
      )}
      {labels[status]}
    </Badge>
  );
}
