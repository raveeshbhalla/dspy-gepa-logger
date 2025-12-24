import { Card, CardContent } from "@/components/ui/card";

type StatsCardsProps = {
  iterations: number;
  candidates: number;
  lmCalls: number;
  evaluations: number;
  seedScore: number | null;
  bestScore: number | null;
};

export function StatsCards({
  iterations,
  candidates,
  lmCalls,
  evaluations,
  seedScore,
  bestScore,
}: StatsCardsProps) {
  const lift = seedScore !== null && bestScore !== null
    ? ((bestScore - seedScore) / Math.max(seedScore, 0.001)) * 100
    : null;

  const stats = [
    { label: "Iterations", value: iterations },
    { label: "Candidates", value: candidates },
    { label: "LM Calls", value: lmCalls.toLocaleString() },
    { label: "Evaluations", value: evaluations.toLocaleString() },
    {
      label: "Seed Score",
      value: seedScore !== null ? `${(seedScore * 100).toFixed(1)}%` : "-",
    },
    {
      label: "Best Score",
      value: bestScore !== null ? `${(bestScore * 100).toFixed(1)}%` : "-",
      highlight: true,
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {stats.map((stat) => (
        <Card key={stat.label} className={stat.highlight ? "border-primary/50" : ""}>
          <CardContent className="p-4">
            <p className="text-2xl font-semibold text-foreground">{stat.value}</p>
            <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
          </CardContent>
        </Card>
      ))}
      {lift !== null && (
        <Card className={lift > 0 ? "border-green-500/50 bg-green-500/5" : lift < 0 ? "border-red-500/50 bg-red-500/5" : ""}>
          <CardContent className="p-4">
            <p className={`text-2xl font-semibold ${lift > 0 ? "text-green-500" : lift < 0 ? "text-red-500" : "text-foreground"}`}>
              {lift > 0 ? "+" : ""}{lift.toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground mt-1">Lift</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
