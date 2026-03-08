import { notFound } from "next/navigation";
import { getClient } from "@/lib/db";
import { getRun, getMetrics, listCheckpoints, getSamples } from "@alpha/db";
import { ModelReport } from "@/components/model-report";

export const dynamic = "force-dynamic";

export default async function ReportPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const client = await getClient();
  const run = await getRun(client, id);
  if (!run) notFound();

  const [metrics, checkpoints, samples] = await Promise.all([
    getMetrics(client, id),
    listCheckpoints(client, id),
    getSamples(client, id),
  ]);

  return (
    <ModelReport
      run={run as any}
      metrics={metrics as any}
      checkpoints={checkpoints as any}
      samples={samples as any}
    />
  );
}
