import Link from "next/link";
import { notFound } from "next/navigation";
import { getCorpusReader } from "@/lib/corpus";
import { ReviewWorkspace } from "./review-workspace";

export const dynamic = "force-dynamic";
export const revalidate = 0;
export const runtime = "nodejs";

export default async function ReviewSessionPage({
  params
}: {
  params: Promise<{ sessionId: string }>;
}) {
  const { sessionId } = await params;
  const loaded = getCorpusReader().reviewPacket(sessionId);
  if (!loaded) notFound();

  return (
    <main className="mx-auto w-full max-w-7xl space-y-5">
      <nav aria-label="Breadcrumb" className="text-xs text-text-muted">
        <Link href="/corpus" className="hover:text-text-primary hover:underline">Alpha Corpus</Link>
        <span className="mx-2">/</span>
        <Link href="/corpus/review" className="hover:text-text-primary hover:underline">Human review</Link>
        <span className="mx-2">/</span>
        <span>Pass {loaded.packet.pass}</span>
      </nav>
      <ReviewWorkspace
        sourcePacket={loaded.packet}
        packetSha256={loaded.packetSha256}
        exportedAt={loaded.exportedAt}
      />
    </main>
  );
}
