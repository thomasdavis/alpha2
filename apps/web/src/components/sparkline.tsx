"use client";

import { useEffect, useState } from "react";
import { Sparkline as UISparkline } from "@alpha/ui";

export function Sparkline({
  runId,
  status,
}: {
  runId: string;
  status: string;
}) {
  const [data, setData] = useState<number[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    fetch(`/api/runs/${encodeURIComponent(runId)}/metrics?last=60`)
      .then((r) => r.json())
      .then((metrics: Array<{ loss: number }>) => {
        if (metrics.length === 0) return;
        setData(metrics.map((m) => m.loss));
        setLoaded(true);
      })
      .catch(() => {});
  }, [runId]);

  const variantMap: Record<string, "success" | "blue" | "warning" | "danger"> = {
    active: "success",
    completed: "blue",
    stale: "warning",
    failed: "danger",
  };

  return (
    <div style={{ opacity: loaded ? 1 : 0.3 }} className="transition-opacity duration-300">
      <UISparkline 
        data={data} 
        variant={variantMap[status] || "default"} 
        className="h-8 w-28 shrink-0 self-center"
      />
    </div>
  );
}
