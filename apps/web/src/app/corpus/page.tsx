import { notFound } from "next/navigation";
import { CorpusExplorer, type CorpusStageCount } from "./corpus-explorer";
import { corpusDatabaseUpdatedAt, getCorpusReader } from "@/lib/corpus";

export const dynamic = "force-dynamic";
export const revalidate = 0;
export const runtime = "nodejs";

type SearchValue = string | string[] | undefined;
type CorpusSearchParams = Promise<Record<string, SearchValue>>;

function one(value: SearchValue): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}
function positiveInteger(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

const stageRelations: { relation: string; label: string; description: string }[] = [
  { relation: "candidate", label: "Candidates", description: "Generated records, including rejected material" },
  { relation: "review", label: "Reviews", description: "Recorded human or model review acts" },
  { relation: "public_training_candidate", label: "Public-ready", description: "Candidates satisfying the current release view" },
  { relation: "release_member", label: "Released", description: "Immutable dataset release membership" },
  { relation: "training_exposure", label: "Exposed", description: "Rows actually shown to a training run" }
];

export default async function CorpusPage({ searchParams }: { searchParams: CorpusSearchParams }) {
  const params = await searchParams;
  const reader = getCorpusReader();
  const relations = reader.listRelations();
  if (relations.length === 0) {
    return (
      <section className="rounded-lg border border-border bg-surface px-6 py-16 text-center">
        <h1 className="text-lg font-semibold text-text-primary">The corpus ledger has no relations yet</h1>
        <p className="mx-auto mt-2 max-w-xl text-sm text-text-secondary">
          This public explorer is connected in read-only mode. Tables and views will appear here as soon as the
          ledger contains them.
        </p>
      </section>
    );
  }

  const requestedRelation = one(params["relation"]);
  const defaultRelation = relations.find((relation) => relation.name === "candidate") ?? relations[0]!;
  const relation = requestedRelation
    ? relations.find((candidate) => candidate.name === requestedRelation)
    : defaultRelation;
  if (!relation) notFound();

  const requestedView = one(params["view"]);
  const view = requestedView === "schema" || requestedView === "lineage" ? requestedView : "rows";
  const direction = one(params["direction"]) === "desc" ? "desc" : "asc";
  const detail = reader.relation(relation.name);
  const page = reader.page(relation.name, {
    page: positiveInteger(one(params["page"]), 1),
    pageSize: positiveInteger(one(params["pageSize"]), 25),
    query: one(params["q"]) ?? "",
    sortColumn: one(params["sort"]),
    sortDirection: direction
  });

  const relationNames = new Set(relations.map((entry) => entry.name));
  const stages: CorpusStageCount[] = stageRelations
    .filter((stage) => relationNames.has(stage.relation))
    .map((stage) => ({
      ...stage,
      count: reader.page(stage.relation, { pageSize: 10 }).totalRows
    }));

  return (
    <CorpusExplorer
      relations={relations}
      detail={detail}
      page={page}
      view={view}
      stages={stages}
      databaseUpdatedAt={corpusDatabaseUpdatedAt()}
      safety={reader.safety()}
    />
  );
}
