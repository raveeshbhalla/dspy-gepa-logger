// Server-side SSE event emitter
// Maintains a registry of connected clients per run

type RunUpdateEvent = {
  type: "iteration" | "evaluation" | "evaluation_feedback" | "candidate" | "lm_call" | "status" | "summary" | "log";
  data: Record<string, unknown>;
};

type GlobalEvent = {
  type: "run_created" | "run_updated" | "run_completed";
  data: Record<string, unknown>;
};

type ClientController = {
  controller: ReadableStreamDefaultController<Uint8Array>;
  runId: string;
};

type GlobalClientController = {
  controller: ReadableStreamDefaultController<Uint8Array>;
};

// Map of runId -> Set of client controllers
const clients = new Map<string, Set<ClientController>>();

// Global clients for sidebar updates (notified of new runs, status changes)
const globalClients = new Set<GlobalClientController>();

export function registerClient(
  runId: string,
  controller: ReadableStreamDefaultController<Uint8Array>
): () => void {
  const clientController: ClientController = { controller, runId };

  if (!clients.has(runId)) {
    clients.set(runId, new Set());
  }
  clients.get(runId)!.add(clientController);

  // Return cleanup function
  return () => {
    const runClients = clients.get(runId);
    if (runClients) {
      runClients.delete(clientController);
      if (runClients.size === 0) {
        clients.delete(runId);
      }
    }
  };
}

export function emitRunUpdate(runId: string, event: RunUpdateEvent): void {
  const runClients = clients.get(runId);
  if (!runClients || runClients.size === 0) {
    return;
  }

  const encoder = new TextEncoder();
  const message = `event: ${event.type}\ndata: ${JSON.stringify(event.data)}\n\n`;
  const encoded = encoder.encode(message);

  for (const client of runClients) {
    try {
      client.controller.enqueue(encoded);
    } catch {
      // Client disconnected, will be cleaned up
      runClients.delete(client);
    }
  }
}

export function getConnectedClientCount(runId: string): number {
  return clients.get(runId)?.size ?? 0;
}

// Global client registration for sidebar updates
export function registerGlobalClient(
  controller: ReadableStreamDefaultController<Uint8Array>
): () => void {
  const clientController: GlobalClientController = { controller };
  globalClients.add(clientController);

  return () => {
    globalClients.delete(clientController);
  };
}

// Emit events to all global clients (for sidebar updates)
export function emitGlobalEvent(event: GlobalEvent): void {
  if (globalClients.size === 0) {
    return;
  }

  const encoder = new TextEncoder();
  const message = `event: ${event.type}\ndata: ${JSON.stringify(event.data)}\n\n`;
  const encoded = encoder.encode(message);

  for (const client of globalClients) {
    try {
      client.controller.enqueue(encoded);
    } catch {
      // Client disconnected, will be cleaned up
      globalClients.delete(client);
    }
  }
}

export function getGlobalClientCount(): number {
  return globalClients.size;
}
