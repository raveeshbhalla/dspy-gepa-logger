import { Card, CardContent } from "@/components/ui/card";

type StatsCardsProps = {
  iterations: number;
  candidates: number;
  lmCalls: number;
  evaluations: number;
  avgSeedScore: number | null;
  avgBestScore: number | null;
  avgImprovement: number | null;
};

export function StatsCards({
  iterations,
  candidates,
  lmCalls,
  evaluations,
  avgSeedScore,
  avgBestScore,
  avgImprovement,
}: StatsCardsProps) {
  const stats = [
    { label: "Iterations", value: iterations },
    { label: "Candidates", value: candidates },
    { label: "LM Calls", value: lmCalls.toLocaleString() },
    { label: "Evaluations", value: evaluations.toLocaleString() },
    {
      label: "Avg Seed Score",
      value: avgSeedScore !== null ? `${(avgSeedScore * 100).toFixed(1)}%` : "-",
    },
    {
      label: "Avg Best Score",
      value: avgBestScore !== null ? `${(avgBestScore * 100).toFixed(1)}%` : "-",
      highlight: true,
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-7 gap-4">
      {stats.map((stat) => (
        <Card key={stat.label} className={stat.highlight ? "border-primary/50" : ""}>
          <CardContent className="p-4">
            <p className="text-2xl font-semibold text-foreground">{stat.value}</p>
            <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
          </CardContent>
        </Card>
      ))}
      {avgImprovement !== null && (
        <Card className={avgImprovement > 0 ? "border-green-500/50 bg-green-500/5" : avgImprovement < 0 ? "border-red-500/50 bg-red-500/5" : ""}>
          <CardContent className="p-4">
            <p className={`text-2xl font-semibold ${avgImprovement > 0 ? "text-green-500" : avgImprovement < 0 ? "text-red-500" : "text-foreground"}`}>
              {avgImprovement > 0 ? "+" : ""}{(avgImprovement * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground mt-1">Avg Improvement</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
