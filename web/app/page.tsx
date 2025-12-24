export default function Home() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-foreground mb-2">
          Welcome to GEPA Logger
        </h2>
        <p className="text-muted-foreground max-w-md">
          Select a run from the sidebar to view its details, or start a new
          optimization with server tracking enabled.
        </p>
        <div className="mt-8 p-4 bg-card rounded-lg border border-border text-left max-w-lg mx-auto">
          <p className="text-sm text-muted-foreground mb-2">Quick start:</p>
          <pre className="text-xs bg-background p-3 rounded overflow-x-auto">
            <code>{`from dspy_gepa_logger import create_logged_gepa

gepa, tracker, _ = create_logged_gepa(
    metric=my_metric,
    server_url="http://localhost:3000",
    project_name="My Project",
)

result = gepa.compile(student, trainset, valset)
tracker.finalize()`}</code>
          </pre>
        </div>
      </div>
    </div>
  );
}
