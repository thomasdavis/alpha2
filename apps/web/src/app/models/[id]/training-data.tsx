"use client";

import { useEffect, useState } from "react";

interface TrainingData {
  text: string;
  bytes: number;
  lines: number;
}

export function TrainingDataPreview({ runId }: { runId: string }) {
  const [data, setData] = useState<TrainingData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch(`/api/runs/${encodeURIComponent(runId)}/training-data`)
      .then((r) => {
        if (!r.ok) throw new Error("not found");
        return r.json();
      })
      .then((d: TrainingData) => { setData(d); setLoading(false); })
      .catch(() => { setError(true); setLoading(false); });
  }, [runId]);

  if (loading) {
    return <p className="text-xs text-text-muted">Loading training data...</p>;
  }

  if (error || !data) {
    return <p className="text-xs text-text-muted">No training data available for this model.</p>;
  }

  const preview = data.text.slice(0, 5000);
  const chars = data.text.length;

  return (
    <div>
      <div className="mb-2 flex items-center gap-3 text-[0.7rem] text-text-muted">
        <span>{chars.toLocaleString()} chars</span>
        <span>&middot;</span>
        <span>{data.lines.toLocaleString()} lines</span>
        <span>&middot;</span>
        <span>{(data.bytes / 1024).toFixed(1)} KB</span>
        {chars > 5000 && (
          <>
            <span>&middot;</span>
            <span className="text-yellow">showing first 5,000 chars</span>
          </>
        )}
      </div>
      <pre className="max-h-80 overflow-auto rounded bg-[#0d0d0d] p-3 text-[0.7rem] leading-relaxed text-text-secondary whitespace-pre-wrap break-words">
        {preview}
      </pre>
    </div>
  );
}
