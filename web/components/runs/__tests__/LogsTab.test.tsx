import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { LogsTab } from "../LogsTab";

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock EventSource
class MockEventSource {
  static instances: MockEventSource[] = [];
  url: string;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  listeners: Record<string, ((event: MessageEvent) => void)[]> = {};

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: (event: MessageEvent) => void) {
    if (!this.listeners[type]) {
      this.listeners[type] = [];
    }
    this.listeners[type].push(listener);
  }

  close() {
    // Clean up
  }

  static clear() {
    MockEventSource.instances = [];
  }
}

// @ts-expect-error - Mocking EventSource
global.EventSource = MockEventSource;

describe("LogsTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    MockEventSource.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("clears logs when runId changes", async () => {
    // First render with run-1
    const logsForRun1 = [
      { id: "1", logType: "stdout", content: "Log from run 1", timestamp: 1000 },
    ];
    const logsForRun2 = [
      { id: "2", logType: "stdout", content: "Log from run 2", timestamp: 2000 },
    ];

    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(logsForRun1),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(logsForRun2),
      });

    const { rerender } = render(<LogsTab runId="run-1" isRunning={false} />);

    // Wait for first logs to appear
    await waitFor(() => {
      expect(screen.getByText("Log from run 1")).toBeInTheDocument();
    });

    // Rerender with new runId
    rerender(<LogsTab runId="run-2" isRunning={false} />);

    // Old logs should be cleared immediately (before new fetch completes)
    // The "No logs yet" message should briefly appear or the new logs should load
    await waitFor(() => {
      expect(screen.queryByText("Log from run 1")).not.toBeInTheDocument();
    });

    // New logs should appear
    await waitFor(() => {
      expect(screen.getByText("Log from run 2")).toBeInTheDocument();
    });
  });

  it("fetches logs for the correct runId", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve([]),
    });

    render(<LogsTab runId="test-run-123" isRunning={false} />);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith("/api/runs/test-run-123/logs");
    });
  });

  it("shows 'No logs yet' when logs array is empty", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve([]),
    });

    render(<LogsTab runId="empty-run" isRunning={false} />);

    await waitFor(() => {
      expect(screen.getByText(/No logs yet/)).toBeInTheDocument();
    });
  });

  it("sets up SSE connection when isRunning is true", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve([]),
    });

    render(<LogsTab runId="running-run" isRunning={true} />);

    await waitFor(() => {
      expect(MockEventSource.instances.length).toBe(1);
      expect(MockEventSource.instances[0].url).toBe("/api/events/running-run");
    });
  });

  it("does not set up SSE connection when isRunning is false", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve([]),
    });

    render(<LogsTab runId="stopped-run" isRunning={false} />);

    // Give it a moment to potentially set up SSE
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(MockEventSource.instances.length).toBe(0);
  });
});
