import { createHash } from "node:crypto";

export interface Migration {
  version: number;
  name: string;
  statements: string[];
}

const immutableTables = [
  "program_version",
  "decision",
  "gate_decision",
  "blob_location",
  "artifact_relation",
  "category_version",
  "category_relation",
  "annotation_assignment",
  "annotation_disagreement",
  "family_version",
  "family_projection",
  "family_projection_relation",
  "competency_question",
  "scene_version",
  "trajectory_member",
  "branch_point",
  "transformation_edge",
  "commitment",
  "commitment_dependency",
  "expected_delta",
  "admissible_analysis_set",
  "admissible_analysis_member",
  "shortcut_hazard",
  "source_version",
  "source_fragment",
  "evidence_anchor",
  "dialogue_version",
  "message_version",
  "participant",
  "utterance_relation",
  "local_term_version",
  "dialogue_state_link",
  "response_policy_target",
  "model_revision",
  "model_role_profile",
  "prompt_template_version",
  "tool_schema",
  "generation_recipe",
  "software_revision",
  "generation_batch",
  "model_call",
  "model_call_message",
  "model_call_tool",
  "model_call_usage",
  "model_call_attempt",
  "raw_artifact",
  "routing_decision",
  "budget_event",
  "candidate_version",
  "candidate_parent",
  "candidate_failure",
  "review",
  "review_dimension_score",
  "review_finding",
  "adjudication",
  "adjudication_basis",
  "disagreement_case",
  "repair_request",
  "quality_state_transition",
  "cohort_snapshot",
  "cohort_member",
  "release_member",
  "release_exclusion",
  "render_job",
  "rendered_unit",
  "rendered_message_map",
  "export_artifact",
  "export_validation",
  "training_exposure",
  "evaluation_output",
  "event",
  "event_object",
  "validation_run",
  "validation_finding"
];

function immutabilityTriggers(table: string): string[] {
  return [
    `CREATE TRIGGER IF NOT EXISTS no_update_${table}
      BEFORE UPDATE ON ${table}
      BEGIN SELECT RAISE(ABORT, '${table} is append-only'); END`,
    `CREATE TRIGGER IF NOT EXISTS no_delete_${table}
      BEFORE DELETE ON ${table}
      BEGIN SELECT RAISE(ABORT, '${table} is append-only'); END`
  ];
}

const v1: string[] = [
  `CREATE TABLE IF NOT EXISTS program (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS program_version (
    id TEXT PRIMARY KEY,
    program_id TEXT NOT NULL REFERENCES program(id),
    version INTEGER NOT NULL,
    objective TEXT NOT NULL,
    authority TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(program_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS decision (
    id TEXT PRIMARY KEY,
    program_id TEXT NOT NULL REFERENCES program(id),
    supersedes_id TEXT REFERENCES decision(id),
    title TEXT NOT NULL,
    decision_text TEXT NOT NULL,
    authority TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS stage_gate (
    id TEXT PRIMARY KEY,
    program_id TEXT NOT NULL REFERENCES program(id),
    slug TEXT NOT NULL,
    definition TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(program_id, slug)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS gate_decision (
    id TEXT PRIMARY KEY,
    stage_gate_id TEXT NOT NULL REFERENCES stage_gate(id),
    outcome TEXT NOT NULL,
    authority TEXT NOT NULL,
    rationale TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS actor (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('human','model','service','organization')),
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS blob (
    sha256 TEXT PRIMARY KEY,
    byte_length INTEGER NOT NULL CHECK(byte_length >= 0),
    media_type TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS blob_location (
    id TEXT PRIMARY KEY,
    blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    location TEXT NOT NULL,
    storage_kind TEXT NOT NULL,
    verified_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(blob_sha256, location)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS artifact_relation (
    id TEXT PRIMARY KEY,
    subject_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    relation TEXT NOT NULL,
    object_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS category (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS category_version (
    id TEXT PRIMARY KEY,
    category_id TEXT NOT NULL REFERENCES category(id),
    version INTEGER NOT NULL,
    preferred_name TEXT NOT NULL,
    concise_definition TEXT NOT NULL,
    extended_definition TEXT NOT NULL,
    meta_class TEXT NOT NULL CHECK(meta_class IN ('design','ground_truth','observed','derived','crosscutting')),
    authority_kind TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(category_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS category_relation (
    id TEXT PRIMARY KEY,
    subject_category_id TEXT NOT NULL REFERENCES category(id),
    relation TEXT NOT NULL,
    object_category_id TEXT NOT NULL REFERENCES category(id),
    scope TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(subject_category_id, relation, object_category_id, scope)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS category_proposal (
    id TEXT PRIMARY KEY,
    proposed_slug TEXT NOT NULL,
    proposed_name TEXT NOT NULL,
    recurring_phenomenon TEXT NOT NULL,
    boundary_argument TEXT NOT NULL,
    nearest_categories_json TEXT NOT NULL,
    competency_questions_json TEXT NOT NULL,
    status TEXT NOT NULL,
    proposer_actor_id TEXT REFERENCES actor(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS annotation_dimension (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    meta_class TEXT NOT NULL,
    definition TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS annotation_assignment (
    id TEXT PRIMARY KEY,
    dimension_id TEXT NOT NULL REFERENCES annotation_dimension(id),
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    value_json TEXT NOT NULL,
    authority_kind TEXT NOT NULL,
    reviewer_actor_id TEXT REFERENCES actor(id),
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS annotation_disagreement (
    id TEXT PRIMARY KEY,
    left_assignment_id TEXT NOT NULL REFERENCES annotation_assignment(id),
    right_assignment_id TEXT NOT NULL REFERENCES annotation_assignment(id),
    relation TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS concept_family (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    split TEXT NOT NULL CHECK(split IN ('unassigned','train','development','public_eval','private_eval','quarantine')),
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS family_version (
    id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES concept_family(id),
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    purpose TEXT NOT NULL,
    blueprint_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    authority_kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(family_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS family_category (
    family_id TEXT NOT NULL REFERENCES concept_family(id),
    category_id TEXT NOT NULL REFERENCES category(id),
    assignment_kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(family_id, category_id, assignment_kind)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS family_projection (
    id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES concept_family(id),
    slug TEXT NOT NULL,
    domain TEXT NOT NULL,
    description TEXT NOT NULL,
    relation TEXT NOT NULL CHECK(relation IN ('true_bridge','false_bridge','partial_bridge')),
    created_at TEXT NOT NULL,
    UNIQUE(family_id, slug)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS family_projection_relation (
    id TEXT PRIMARY KEY,
    subject_projection_id TEXT NOT NULL REFERENCES family_projection(id),
    relation TEXT NOT NULL,
    object_projection_id TEXT NOT NULL REFERENCES family_projection(id),
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS competency_question (
    id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES concept_family(id),
    question_text TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS scene (
    id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES concept_family(id),
    scene_key TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(family_id, scene_key)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS scene_version (
    id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL REFERENCES scene(id),
    version INTEGER NOT NULL,
    purpose TEXT NOT NULL,
    scene_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(scene_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS trajectory (
    id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES concept_family(id),
    slug TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(family_id, slug)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS trajectory_member (
    id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL REFERENCES trajectory(id),
    scene_id TEXT NOT NULL REFERENCES scene(id),
    ordinal INTEGER NOT NULL,
    branch_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(trajectory_id, ordinal, branch_key)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS branch_point (
    id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL REFERENCES trajectory(id),
    after_ordinal INTEGER NOT NULL,
    branch_key TEXT NOT NULL,
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(trajectory_id, after_ordinal, branch_key)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS transformation (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    definition TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS transformation_edge (
    id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES concept_family(id),
    source_key TEXT NOT NULL,
    target_key TEXT NOT NULL,
    transformation_id TEXT NOT NULL REFERENCES transformation(id),
    expected_delta_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS semantic_state (
    id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES concept_family(id),
    state_key TEXT NOT NULL,
    purpose TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(family_id, state_key)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS commitment (
    id TEXT PRIMARY KEY,
    semantic_state_id TEXT NOT NULL REFERENCES semantic_state(id),
    holder TEXT NOT NULL,
    proposition TEXT NOT NULL,
    status TEXT NOT NULL,
    scope TEXT NOT NULL,
    depends_on_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS commitment_dependency (
    id TEXT PRIMARY KEY,
    dependent_commitment_id TEXT NOT NULL REFERENCES commitment(id),
    dependency_kind TEXT NOT NULL,
    dependency_subject_kind TEXT NOT NULL,
    dependency_subject_id TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS expected_delta (
    id TEXT PRIMARY KEY,
    transformation_edge_id TEXT REFERENCES transformation_edge(id),
    candidate_key TEXT,
    operation TEXT NOT NULL CHECK(operation IN ('preserve','add','retract','pluralize','attribute','temporalize','unsupported')),
    proposition TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS admissible_analysis_set (
    id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES concept_family(id),
    set_key TEXT NOT NULL,
    plurality_kind TEXT NOT NULL,
    clarification_can_reduce INTEGER NOT NULL CHECK(clarification_can_reduce IN (0,1)),
    created_at TEXT NOT NULL,
    UNIQUE(family_id, set_key)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS admissible_analysis_member (
    id TEXT PRIMARY KEY,
    analysis_set_id TEXT NOT NULL REFERENCES admissible_analysis_set(id),
    status TEXT NOT NULL CHECK(status IN ('required','permitted','excluded')),
    analysis_text TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS shortcut_hazard (
    id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES concept_family(id),
    hazard_kind TEXT NOT NULL,
    description TEXT NOT NULL,
    detection_plan TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS source (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    synthetic INTEGER NOT NULL CHECK(synthetic IN (0,1)),
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS source_version (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES source(id),
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    citation TEXT NOT NULL,
    content_blob_sha256 TEXT REFERENCES blob(sha256),
    license_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(source_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS source_fragment (
    id TEXT PRIMARY KEY,
    source_version_id TEXT NOT NULL REFERENCES source_version(id),
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    text_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK(end_offset >= start_offset)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS evidence_anchor (
    id TEXT PRIMARY KEY,
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragment(id),
    relation TEXT NOT NULL,
    anchoring_method TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS dialogue (
    id TEXT PRIMARY KEY,
    family_id TEXT REFERENCES concept_family(id),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS dialogue_version (
    id TEXT PRIMARY KEY,
    dialogue_id TEXT NOT NULL REFERENCES dialogue(id),
    version INTEGER NOT NULL,
    purpose TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(dialogue_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS participant (
    id TEXT PRIMARY KEY,
    dialogue_id TEXT NOT NULL REFERENCES dialogue(id),
    participant_key TEXT NOT NULL,
    role TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(dialogue_id, participant_key)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS message (
    id TEXT PRIMARY KEY,
    dialogue_id TEXT NOT NULL REFERENCES dialogue(id),
    ordinal INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(dialogue_id, ordinal)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS message_version (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES message(id),
    version INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
    natural_text_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    language TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(message_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS utterance_relation (
    id TEXT PRIMARY KEY,
    subject_message_id TEXT NOT NULL REFERENCES message(id),
    relation TEXT NOT NULL,
    object_message_id TEXT NOT NULL REFERENCES message(id),
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS local_term (
    id TEXT PRIMARY KEY,
    dialogue_id TEXT NOT NULL REFERENCES dialogue(id),
    term_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(dialogue_id, term_key)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS local_term_version (
    id TEXT PRIMARY KEY,
    local_term_id TEXT NOT NULL REFERENCES local_term(id),
    version INTEGER NOT NULL,
    surface_form TEXT NOT NULL,
    definition TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(local_term_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS dialogue_state_link (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES message(id),
    state_before_id TEXT REFERENCES semantic_state(id),
    state_after_id TEXT REFERENCES semantic_state(id),
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS response_policy_target (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES message(id),
    policy_slug TEXT NOT NULL,
    necessity TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS model (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(provider, model_id)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS provider (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    transport TEXT NOT NULL,
    terms_snapshot_blob_sha256 TEXT REFERENCES blob(sha256),
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS model_revision (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model(id),
    revision TEXT NOT NULL,
    role TEXT NOT NULL,
    transport TEXT NOT NULL,
    cli_version TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(model_id, revision, role, transport)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS model_role_profile (
    id TEXT PRIMARY KEY,
    model_revision_id TEXT NOT NULL REFERENCES model_revision(id),
    task_class TEXT NOT NULL,
    eligibility TEXT NOT NULL,
    calibration_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(model_revision_id, task_class)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS prompt_template (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS prompt_template_version (
    id TEXT PRIMARY KEY,
    prompt_template_id TEXT NOT NULL REFERENCES prompt_template(id),
    version INTEGER NOT NULL,
    content_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(prompt_template_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS tool_schema (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    version INTEGER NOT NULL,
    schema_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(slug, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS generation_recipe (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    version INTEGER NOT NULL,
    model_revision_id TEXT NOT NULL REFERENCES model_revision(id),
    prompt_template_version_id TEXT NOT NULL REFERENCES prompt_template_version(id),
    tool_schema_id TEXT NOT NULL REFERENCES tool_schema(id),
    config_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(slug, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS software_revision (
    id TEXT PRIMARY KEY,
    component TEXT NOT NULL,
    revision TEXT NOT NULL,
    build_digest TEXT,
    environment_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(component, revision, environment_json)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS generation_campaign (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    purpose TEXT NOT NULL,
    status TEXT NOT NULL,
    worker_model TEXT NOT NULL,
    critic_model TEXT NOT NULL,
    max_generation_calls INTEGER NOT NULL,
    max_review_calls INTEGER NOT NULL,
    items_per_family INTEGER NOT NULL,
    artifact_limit_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS generation_task (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES generation_campaign(id),
    family_id TEXT REFERENCES concept_family(id),
    task_kind TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS generation_batch (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES generation_campaign(id),
    batch_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    recipe_id TEXT REFERENCES generation_recipe(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(campaign_id, batch_index)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS model_call (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES generation_task(id),
    model_revision_id TEXT NOT NULL REFERENCES model_revision(id),
    prompt_template_version_id TEXT REFERENCES prompt_template_version(id),
    tool_schema_id TEXT REFERENCES tool_schema(id),
    request_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    stdout_blob_sha256 TEXT REFERENCES blob(sha256),
    stderr_blob_sha256 TEXT REFERENCES blob(sha256),
    response_blob_sha256 TEXT REFERENCES blob(sha256),
    command_json TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    input_tokens INTEGER,
    cached_input_tokens INTEGER,
    output_tokens INTEGER,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS model_call_message (
    id TEXT PRIMARY KEY,
    model_call_id TEXT NOT NULL REFERENCES model_call(id),
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL,
    content_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    created_at TEXT NOT NULL,
    UNIQUE(model_call_id, ordinal)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS model_call_tool (
    id TEXT PRIMARY KEY,
    model_call_id TEXT NOT NULL REFERENCES model_call(id),
    tool_schema_id TEXT REFERENCES tool_schema(id),
    tool_name TEXT NOT NULL,
    input_blob_sha256 TEXT REFERENCES blob(sha256),
    output_blob_sha256 TEXT REFERENCES blob(sha256),
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS model_call_usage (
    id TEXT PRIMARY KEY,
    model_call_id TEXT NOT NULL REFERENCES model_call(id),
    input_tokens INTEGER,
    cached_input_tokens INTEGER,
    output_tokens INTEGER,
    monetary_cost REAL,
    usage_source TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS model_call_attempt (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES generation_task(id),
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    error_text TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    UNIQUE(task_id, attempt)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS raw_artifact (
    id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES generation_task(id),
    kind TEXT NOT NULL,
    blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS routing_decision (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES generation_task(id),
    selected_model_revision_id TEXT NOT NULL REFERENCES model_revision(id),
    rationale TEXT NOT NULL,
    alternatives_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS budget_event (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES generation_campaign(id),
    task_id TEXT REFERENCES generation_task(id),
    event_kind TEXT NOT NULL,
    calls_delta INTEGER NOT NULL,
    token_delta INTEGER,
    monetary_delta REAL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS generation_event (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES generation_campaign(id),
    task_id TEXT REFERENCES generation_task(id),
    event_kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS candidate (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES generation_campaign(id),
    family_id TEXT NOT NULL REFERENCES concept_family(id),
    item_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(campaign_id, family_id, item_key)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS candidate_version (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    version INTEGER NOT NULL,
    generation_call_id TEXT REFERENCES model_call(id),
    dialogue_id TEXT REFERENCES dialogue(id),
    content_json TEXT NOT NULL,
    hidden_contract_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(candidate_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS candidate_parent (
    id TEXT PRIMARY KEY,
    child_candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    parent_candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS candidate_failure (
    id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES generation_task(id),
    candidate_id TEXT REFERENCES candidate(id),
    code TEXT NOT NULL,
    detail TEXT NOT NULL,
    raw_artifact_id TEXT REFERENCES raw_artifact(id),
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS review (
    id TEXT PRIMARY KEY,
    candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    reviewer_model_revision_id TEXT REFERENCES model_revision(id),
    reviewer_actor_id TEXT REFERENCES actor(id),
    review_call_id TEXT REFERENCES model_call(id),
    outcome TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS rubric (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS rubric_version (
    id TEXT PRIMARY KEY,
    rubric_id TEXT NOT NULL REFERENCES rubric(id),
    version INTEGER NOT NULL,
    definition_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(rubric_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS review_assignment (
    id TEXT PRIMARY KEY,
    candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    reviewer_model_revision_id TEXT REFERENCES model_revision(id),
    reviewer_actor_id TEXT REFERENCES actor(id),
    rubric_version_id TEXT REFERENCES rubric_version(id),
    blindness_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS review_dimension_score (
    id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL REFERENCES review(id),
    dimension TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(review_id, dimension)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS review_finding (
    id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL REFERENCES review(id),
    dimension TEXT NOT NULL,
    severity TEXT NOT NULL,
    evidence TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS adjudication (
    id TEXT PRIMARY KEY,
    candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    authority TEXT NOT NULL,
    outcome TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS adjudication_basis (
    id TEXT PRIMARY KEY,
    adjudication_id TEXT NOT NULL REFERENCES adjudication(id),
    basis_kind TEXT NOT NULL,
    basis_id TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS disagreement_case (
    id TEXT PRIMARY KEY,
    candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    status TEXT NOT NULL,
    description TEXT NOT NULL,
    review_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS repair_request (
    id TEXT PRIMARY KEY,
    candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    review_id TEXT REFERENCES review(id),
    requested_change TEXT NOT NULL,
    preserve_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS quality_state_transition (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    reason TEXT NOT NULL,
    authority TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS cohort_definition (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    version INTEGER NOT NULL,
    query_text TEXT NOT NULL,
    policy_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(slug, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS cohort_snapshot (
    id TEXT PRIMARY KEY,
    cohort_definition_id TEXT NOT NULL REFERENCES cohort_definition(id),
    snapshot_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS cohort_member (
    id TEXT PRIMARY KEY,
    cohort_snapshot_id TEXT NOT NULL REFERENCES cohort_snapshot(id),
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    membership_reason TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS dataset_release (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    manifest_blob_sha256 TEXT REFERENCES blob(sha256),
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS release_member (
    id TEXT PRIMARY KEY,
    release_id TEXT NOT NULL REFERENCES dataset_release(id),
    candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    split TEXT NOT NULL,
    membership_reason TEXT NOT NULL,
    weight REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(release_id, candidate_version_id)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS release_exclusion (
    id TEXT PRIMARY KEY,
    release_id TEXT NOT NULL REFERENCES dataset_release(id),
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    exclusion_reason TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS renderer (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS renderer_version (
    id TEXT PRIMARY KEY,
    renderer_id TEXT NOT NULL REFERENCES renderer(id),
    version INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(renderer_id, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS render_job (
    id TEXT PRIMARY KEY,
    renderer_version_id TEXT NOT NULL REFERENCES renderer_version(id),
    cohort_snapshot_id TEXT REFERENCES cohort_snapshot(id),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS rendered_unit (
    id TEXT PRIMARY KEY,
    candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    renderer_version_id TEXT NOT NULL REFERENCES renderer_version(id),
    rendered_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    token_ids_blob_sha256 TEXT REFERENCES blob(sha256),
    loss_mask_blob_sha256 TEXT REFERENCES blob(sha256),
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(candidate_version_id, renderer_version_id)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS rendered_message_map (
    id TEXT PRIMARY KEY,
    rendered_unit_id TEXT NOT NULL REFERENCES rendered_unit(id),
    message_version_id TEXT NOT NULL REFERENCES message_version(id),
    byte_start INTEGER NOT NULL,
    byte_end INTEGER NOT NULL,
    token_start INTEGER,
    token_end INTEGER,
    created_at TEXT NOT NULL,
    CHECK(byte_end >= byte_start)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS export_artifact (
    id TEXT PRIMARY KEY,
    release_id TEXT REFERENCES dataset_release(id),
    cohort_snapshot_id TEXT REFERENCES cohort_snapshot(id),
    format TEXT NOT NULL,
    blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS export_validation (
    id TEXT PRIMARY KEY,
    export_artifact_id TEXT NOT NULL REFERENCES export_artifact(id),
    validator TEXT NOT NULL,
    outcome TEXT NOT NULL,
    findings_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS training_exposure (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    rendered_unit_id TEXT NOT NULL REFERENCES rendered_unit(id),
    step INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    weight REAL NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS evaluation_output (
    id TEXT PRIMARY KEY,
    checkpoint_id TEXT NOT NULL,
    evaluation_item_id TEXT NOT NULL,
    output_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    decoder_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS event (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS event_object (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES event(id),
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS validation_run (
    id TEXT PRIMARY KEY,
    validator TEXT NOT NULL,
    software_revision_id TEXT REFERENCES software_revision(id),
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS validation_finding (
    id TEXT PRIMARY KEY,
    validation_run_id TEXT NOT NULL REFERENCES validation_run(id),
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    code TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
  ) STRICT`,
  `CREATE INDEX IF NOT EXISTS idx_task_campaign_status ON generation_task(campaign_id, status)`,
  `CREATE INDEX IF NOT EXISTS idx_candidate_campaign_status ON candidate(campaign_id, status)`,
  `CREATE INDEX IF NOT EXISTS idx_candidate_family ON candidate(family_id)`,
  `CREATE INDEX IF NOT EXISTS idx_review_candidate ON review(candidate_version_id)`,
  `CREATE INDEX IF NOT EXISTS idx_message_dialogue ON message(dialogue_id, ordinal)`,
  `CREATE INDEX IF NOT EXISTS idx_event_object ON event(object_kind, object_id, created_at)`,
  ...immutableTables.flatMap(immutabilityTriggers)
];

const v2: string[] = [
  `CREATE VIEW IF NOT EXISTS corpus_candidate_current AS
    SELECT c.id AS candidate_id,
           c.campaign_id,
           c.family_id,
           cf.slug AS family_slug,
           c.item_key,
           c.kind,
           c.status,
           cv.id AS candidate_version_id,
           cv.version,
           cv.content_json,
           cv.hidden_contract_json,
           cv.content_sha256,
           cv.created_at
    FROM candidate c
    JOIN concept_family cf ON cf.id = c.family_id
    JOIN candidate_version cv ON cv.candidate_id = c.id
    WHERE cv.version = (
      SELECT MAX(cv2.version) FROM candidate_version cv2 WHERE cv2.candidate_id = c.id
    )`,
  `CREATE VIEW IF NOT EXISTS public_training_candidate AS
    SELECT candidate_id,
           family_slug,
           item_key,
           kind,
           candidate_version_id,
           version,
           content_json,
           content_sha256,
           created_at
    FROM corpus_candidate_current
    WHERE status = 'human_accepted'`,
  `CREATE VIEW IF NOT EXISTS campaign_progress AS
    SELECT gc.id AS campaign_id,
           gc.slug,
           gc.status,
           COUNT(DISTINCT gt.id) AS task_count,
           COUNT(DISTINCT CASE WHEN gt.status = 'completed' THEN gt.id END) AS completed_tasks,
           COUNT(DISTINCT mc.id) AS model_calls,
           COUNT(DISTINCT c.id) AS candidates,
           COUNT(DISTINCT CASE WHEN c.status = 'structurally_valid' THEN c.id END) AS structurally_valid,
           COUNT(DISTINCT CASE WHEN c.status = 'structurally_rejected' THEN c.id END) AS structurally_rejected,
           COUNT(DISTINCT CASE WHEN c.status = 'human_accepted' THEN c.id END) AS human_accepted
    FROM generation_campaign gc
    LEFT JOIN generation_task gt ON gt.campaign_id = gc.id
    LEFT JOIN model_call mc ON mc.task_id = gt.id
    LEFT JOIN candidate c ON c.campaign_id = gc.id
    GROUP BY gc.id, gc.slug, gc.status`,
  `CREATE VIEW IF NOT EXISTS candidate_review_state AS
    SELECT cc.candidate_id,
           cc.family_slug,
           cc.item_key,
           cc.status,
           COUNT(DISTINCT r.id) AS model_or_human_reviews,
           COUNT(DISTINCT a.id) AS adjudications,
           MAX(r.created_at) AS latest_review_at,
           MAX(a.created_at) AS latest_adjudication_at
    FROM corpus_candidate_current cc
    LEFT JOIN review r ON r.candidate_version_id = cc.candidate_version_id
    LEFT JOIN adjudication a ON a.candidate_version_id = cc.candidate_version_id
    GROUP BY cc.candidate_id, cc.family_slug, cc.item_key, cc.status`
];

const v3ImmutableTables = [
  "analysis_method",
  "analysis_run",
  "analysis_metric",
  "similarity_edge",
  "template_signature"
];

const v3: string[] = [
  `CREATE TABLE IF NOT EXISTS analysis_method (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    version INTEGER NOT NULL,
    definition TEXT NOT NULL,
    config_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(slug, version)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS analysis_run (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES generation_campaign(id),
    analysis_method_id TEXT NOT NULL REFERENCES analysis_method(id),
    software_revision_id TEXT NOT NULL REFERENCES software_revision(id),
    input_snapshot_sha256 TEXT NOT NULL,
    output_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    status TEXT NOT NULL,
    evidence_scope TEXT NOT NULL,
    disclaimer TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    UNIQUE(campaign_id, analysis_method_id, software_revision_id, input_snapshot_sha256),
    CHECK(status = 'completed'),
    CHECK(evidence_scope = 'surface_distribution_only')
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS analysis_metric (
    id TEXT PRIMARY KEY,
    analysis_run_id TEXT NOT NULL REFERENCES analysis_run(id),
    scope_kind TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    value_real REAL,
    value_text TEXT,
    unit TEXT NOT NULL,
    denominator REAL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(analysis_run_id, scope_kind, scope_id, metric),
    CHECK((value_real IS NOT NULL AND value_text IS NULL)
       OR (value_real IS NULL AND value_text IS NOT NULL))
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS similarity_edge (
    id TEXT PRIMARY KEY,
    analysis_run_id TEXT NOT NULL REFERENCES analysis_run(id),
    left_candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    right_candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    method TEXT NOT NULL,
    score REAL NOT NULL,
    review_threshold REAL NOT NULL,
    classification TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(analysis_run_id, left_candidate_version_id, right_candidate_version_id, method),
    CHECK(left_candidate_version_id < right_candidate_version_id),
    CHECK(score >= 0.0 AND score <= 1.0),
    CHECK(review_threshold >= 0.0 AND review_threshold <= 1.0),
    CHECK(classification IN ('surface_review_candidate', 'not_flagged'))
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS template_signature (
    id TEXT PRIMARY KEY,
    analysis_run_id TEXT NOT NULL REFERENCES analysis_run(id),
    scope_kind TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    signature_kind TEXT NOT NULL,
    signature TEXT NOT NULL,
    candidate_count INTEGER NOT NULL,
    denominator INTEGER NOT NULL,
    rate REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(analysis_run_id, scope_kind, scope_id, signature_kind, signature),
    CHECK(candidate_count >= 0),
    CHECK(denominator > 0),
    CHECK(rate >= 0.0 AND rate <= 1.0)
  ) STRICT`,
  `CREATE INDEX IF NOT EXISTS idx_analysis_run_campaign
     ON analysis_run(campaign_id, completed_at)`,
  `CREATE INDEX IF NOT EXISTS idx_analysis_metric_run_scope
     ON analysis_metric(analysis_run_id, scope_kind, scope_id, metric)`,
  `CREATE INDEX IF NOT EXISTS idx_similarity_edge_run_score
     ON similarity_edge(analysis_run_id, method, score DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_template_signature_run_rate
     ON template_signature(analysis_run_id, scope_kind, scope_id, rate DESC)`,
  ...v3ImmutableTables.flatMap(immutabilityTriggers)
];

const v4: string[] = [
  `CREATE TABLE IF NOT EXISTS analysis_run_correction (
    id TEXT PRIMARY KEY,
    erroneous_analysis_run_id TEXT NOT NULL REFERENCES analysis_run(id),
    corrected_analysis_run_id TEXT NOT NULL REFERENCES analysis_run(id),
    reason TEXT NOT NULL,
    authority TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(erroneous_analysis_run_id, corrected_analysis_run_id),
    CHECK(erroneous_analysis_run_id <> corrected_analysis_run_id)
  ) STRICT`,
  `CREATE INDEX IF NOT EXISTS idx_analysis_run_correction_erroneous
     ON analysis_run_correction(erroneous_analysis_run_id, created_at)`,
  ...immutabilityTriggers("analysis_run_correction")
];

const v5ImmutableTables = [
  "family_synthesis",
  "family_synthesis_basis",
  "structural_disposition",
  "structural_disposition_basis"
];

const v5: string[] = [
  `CREATE TABLE IF NOT EXISTS family_synthesis_assignment (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES generation_campaign(id),
    family_version_id TEXT NOT NULL REFERENCES family_version(id),
    reviewer_actor_id TEXT NOT NULL REFERENCES actor(id),
    rubric_version_id TEXT NOT NULL REFERENCES rubric_version(id),
    session_id TEXT NOT NULL,
    input_snapshot_sha256 TEXT NOT NULL,
    blindness_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(campaign_id, family_version_id, reviewer_actor_id, rubric_version_id),
    CHECK(status IN ('assigned', 'completed'))
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS family_synthesis (
    id TEXT PRIMARY KEY,
    assignment_id TEXT NOT NULL UNIQUE REFERENCES family_synthesis_assignment(id),
    family_version_id TEXT NOT NULL REFERENCES family_version(id),
    reviewer_actor_id TEXT NOT NULL REFERENCES actor(id),
    disposition TEXT NOT NULL,
    central_distinction TEXT NOT NULL,
    coverage_json TEXT NOT NULL,
    diagnosis_json TEXT NOT NULL,
    rationale TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    submission_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    created_at TEXT NOT NULL,
    CHECK(confidence BETWEEN 0 AND 4)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS family_synthesis_basis (
    id TEXT PRIMARY KEY,
    family_synthesis_id TEXT NOT NULL REFERENCES family_synthesis(id),
    candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    review_id TEXT NOT NULL REFERENCES review(id),
    review_pass TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(family_synthesis_id, review_id),
    CHECK(review_pass IN ('A', 'B'))
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS structural_disposition (
    id TEXT PRIMARY KEY,
    family_synthesis_id TEXT NOT NULL REFERENCES family_synthesis(id),
    candidate_version_id TEXT NOT NULL REFERENCES candidate_version(id),
    reviewer_actor_id TEXT NOT NULL REFERENCES actor(id),
    content_utility TEXT NOT NULL,
    validator_finding_correctness TEXT NOT NULL,
    identified_value TEXT NOT NULL,
    semantic_type TEXT NOT NULL,
    remedy TEXT NOT NULL,
    automatic_acceptance_hazard TEXT NOT NULL,
    automatic_rejection_hazard TEXT NOT NULL,
    rationale TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(family_synthesis_id, candidate_version_id),
    CHECK(confidence BETWEEN 0 AND 4)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS structural_disposition_basis (
    id TEXT PRIMARY KEY,
    structural_disposition_id TEXT NOT NULL REFERENCES structural_disposition(id),
    basis_kind TEXT NOT NULL,
    basis_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(structural_disposition_id, basis_kind, basis_id)
  ) STRICT`,
  `CREATE INDEX IF NOT EXISTS idx_family_synthesis_assignment_status
     ON family_synthesis_assignment(campaign_id, reviewer_actor_id, status)`,
  `CREATE INDEX IF NOT EXISTS idx_family_synthesis_family
     ON family_synthesis(family_version_id, created_at)`,
  `CREATE INDEX IF NOT EXISTS idx_structural_disposition_candidate
     ON structural_disposition(candidate_version_id, created_at)`,
  ...v5ImmutableTables.flatMap(immutabilityTriggers)
];

const v6ImmutableTables = [
  "review_presentation_response",
  "review_presentation_score",
  "review_presentation_finding"
];

const v6: string[] = [
  `CREATE TABLE IF NOT EXISTS review_presentation_session (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES generation_campaign(id),
    reviewer_actor_id TEXT NOT NULL REFERENCES actor(id),
    rubric_version_id TEXT NOT NULL REFERENCES rubric_version(id),
    review_pass TEXT NOT NULL,
    seed TEXT NOT NULL,
    input_snapshot_sha256 TEXT NOT NULL,
    requested_presentations INTEGER NOT NULL,
    repeat_presentations INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK(review_pass IN ('A', 'B')),
    CHECK(requested_presentations > 0),
    CHECK(repeat_presentations >= 0 AND repeat_presentations <= requested_presentations),
    CHECK(status IN ('assigned', 'completed'))
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS review_presentation (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES review_presentation_session(id),
    review_assignment_id TEXT NOT NULL REFERENCES review_assignment(id),
    presentation_kind TEXT NOT NULL,
    source_review_id TEXT REFERENCES review(id),
    ordinal INTEGER NOT NULL,
    opaque_item_id TEXT NOT NULL,
    candidate_content_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, ordinal),
    UNIQUE(review_assignment_id, presentation_kind),
    CHECK(presentation_kind IN ('primary', 'hidden_repeat')),
    CHECK((presentation_kind = 'primary' AND source_review_id IS NULL)
       OR (presentation_kind = 'hidden_repeat' AND source_review_id IS NOT NULL)),
    CHECK(ordinal > 0),
    CHECK(status IN ('assigned', 'completed'))
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS review_presentation_response (
    id TEXT PRIMARY KEY,
    presentation_id TEXT NOT NULL UNIQUE REFERENCES review_presentation(id),
    reviewer_actor_id TEXT NOT NULL REFERENCES actor(id),
    created_review_id TEXT REFERENCES review(id),
    outcome TEXT NOT NULL,
    response_json TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    submission_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    created_at TEXT NOT NULL,
    CHECK(confidence BETWEEN 0 AND 4)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS review_presentation_score (
    id TEXT PRIMARY KEY,
    presentation_response_id TEXT NOT NULL REFERENCES review_presentation_response(id),
    dimension TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(presentation_response_id, dimension),
    CHECK(score >= 0.0 AND score <= 4.0)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS review_presentation_finding (
    id TEXT PRIMARY KEY,
    presentation_response_id TEXT NOT NULL REFERENCES review_presentation_response(id),
    ordinal INTEGER NOT NULL,
    dimension TEXT NOT NULL,
    severity TEXT NOT NULL,
    evidence TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(presentation_response_id, ordinal),
    CHECK(ordinal > 0),
    CHECK(severity IN ('observation', 'minor', 'major', 'critical'))
  ) STRICT`,
  `CREATE INDEX IF NOT EXISTS idx_review_presentation_session_status
     ON review_presentation_session(campaign_id, reviewer_actor_id, review_pass, status)`,
  `CREATE INDEX IF NOT EXISTS idx_review_presentation_assignment_kind
     ON review_presentation(review_assignment_id, presentation_kind, status)`,
  `CREATE VIEW IF NOT EXISTS review_repeat_stability AS
     SELECT rp.id AS presentation_id,
            rps.id AS session_id,
            rps.campaign_id,
            rp.review_assignment_id,
            ra.candidate_version_id,
            rp.source_review_id,
            rpr.id AS repeat_response_id,
            CASE WHEN rpr.outcome = source.outcome THEN 1 ELSE 0 END AS outcome_match,
            CASE WHEN json_extract(rpr.response_json, '$.questionPolicy')
                       = json_extract(source.rationale, '$.questionPolicy') THEN 1 ELSE 0 END AS question_policy_match,
            CASE WHEN json_extract(rpr.response_json, '$.missingClarification')
                       = json_extract(source.rationale, '$.missingClarification') THEN 1 ELSE 0 END AS missing_clarification_match,
            ABS(rpr.confidence - CAST(json_extract(source.rationale, '$.confidence') AS INTEGER)) AS confidence_delta,
            (SELECT AVG(CASE WHEN rpscore.score = original.score THEN 1.0 ELSE 0.0 END)
               FROM review_presentation_score rpscore
               JOIN review_dimension_score original
                 ON original.review_id = rp.source_review_id
                AND original.dimension = rpscore.dimension
              WHERE rpscore.presentation_response_id = rpr.id) AS dimension_exact_rate,
            (SELECT AVG(ABS(rpscore.score - original.score))
               FROM review_presentation_score rpscore
               JOIN review_dimension_score original
                 ON original.review_id = rp.source_review_id
                AND original.dimension = rpscore.dimension
              WHERE rpscore.presentation_response_id = rpr.id) AS mean_absolute_score_delta,
            rpr.created_at
       FROM review_presentation rp
       JOIN review_presentation_session rps ON rps.id = rp.session_id
       JOIN review_assignment ra ON ra.id = rp.review_assignment_id
       JOIN review_presentation_response rpr ON rpr.presentation_id = rp.id
       JOIN review source ON source.id = rp.source_review_id
      WHERE rp.presentation_kind = 'hidden_repeat'`,
  ...v6ImmutableTables.flatMap(immutabilityTriggers)
];

const v7ImmutableTables = [
  "campaign_closeout",
  "campaign_closeout_state",
  "campaign_closeout_basis",
  "campaign_failure_cluster",
  "campaign_failure_cluster_member",
  "campaign_distribution_assessment"
];

const v7: string[] = [
  `CREATE TABLE IF NOT EXISTS campaign_closeout_assignment (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES generation_campaign(id),
    adjudicator_actor_id TEXT NOT NULL REFERENCES actor(id),
    rubric_version_id TEXT NOT NULL REFERENCES rubric_version(id),
    session_id TEXT NOT NULL,
    input_snapshot_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(campaign_id, adjudicator_actor_id, rubric_version_id),
    CHECK(status IN ('assigned', 'completed'))
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS campaign_closeout (
    id TEXT PRIMARY KEY,
    assignment_id TEXT NOT NULL UNIQUE REFERENCES campaign_closeout_assignment(id),
    campaign_id TEXT NOT NULL REFERENCES generation_campaign(id),
    adjudicator_actor_id TEXT NOT NULL REFERENCES actor(id),
    recommendation_summary TEXT NOT NULL,
    known_json TEXT NOT NULL,
    unknown_json TEXT NOT NULL,
    proposed_next_json TEXT NOT NULL,
    disagreement_json TEXT NOT NULL,
    overall_rationale TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    execution_authorized INTEGER NOT NULL DEFAULT 0,
    submission_blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
    created_at TEXT NOT NULL,
    CHECK(confidence BETWEEN 0 AND 4),
    CHECK(execution_authorized = 0)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS campaign_closeout_state (
    id TEXT PRIMARY KEY,
    campaign_closeout_id TEXT NOT NULL REFERENCES campaign_closeout(id),
    state TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(campaign_closeout_id, state),
    CHECK(state IN ('D5_REPAIR_REQUIRED', 'D5_CRITIC_CALIBRATION_JUSTIFIED',
      'D5_BATCHING_PROBE_JUSTIFIED', 'D5_EVALUATION_DESIGN_JUSTIFIED', 'D5_STOP'))
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS campaign_closeout_basis (
    id TEXT PRIMARY KEY,
    campaign_closeout_id TEXT NOT NULL REFERENCES campaign_closeout(id),
    basis_kind TEXT NOT NULL,
    basis_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(campaign_closeout_id, basis_kind, basis_id)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS campaign_failure_cluster (
    id TEXT PRIMARY KEY,
    campaign_closeout_id TEXT NOT NULL REFERENCES campaign_closeout(id),
    cluster_key TEXT NOT NULL,
    label TEXT NOT NULL,
    locus TEXT NOT NULL,
    severity TEXT NOT NULL,
    proposed_repair TEXT NOT NULL,
    new_calls_needed TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(campaign_closeout_id, cluster_key)
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS campaign_failure_cluster_member (
    id TEXT PRIMARY KEY,
    failure_cluster_id TEXT NOT NULL REFERENCES campaign_failure_cluster(id),
    member_kind TEXT NOT NULL,
    member_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(failure_cluster_id, member_kind, member_id),
    CHECK(member_kind IN ('candidate_version', 'family_version', 'review', 'family_synthesis',
      'structural_disposition'))
  ) STRICT`,
  `CREATE TABLE IF NOT EXISTS campaign_distribution_assessment (
    id TEXT PRIMARY KEY,
    campaign_closeout_id TEXT NOT NULL REFERENCES campaign_closeout(id),
    dimension TEXT NOT NULL,
    assessment TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(campaign_closeout_id, dimension)
  ) STRICT`,
  `CREATE INDEX IF NOT EXISTS idx_campaign_closeout_assignment_status
     ON campaign_closeout_assignment(campaign_id, adjudicator_actor_id, status)`,
  `CREATE INDEX IF NOT EXISTS idx_campaign_closeout_state
     ON campaign_closeout_state(state, created_at)`,
  `CREATE INDEX IF NOT EXISTS idx_campaign_failure_cluster_locus
     ON campaign_failure_cluster(locus, severity)`,
  ...v7ImmutableTables.flatMap(immutabilityTriggers)
];

export const migrations: Migration[] = [
  { version: 1, name: "initial_scientific_ledger", statements: v1 },
  { version: 2, name: "current_and_public_views", statements: v2 },
  { version: 3, name: "first_class_surface_analysis", statements: v3 },
  { version: 4, name: "append_only_analysis_run_corrections", statements: v4 },
  { version: 5, name: "d5_family_synthesis_and_structural_disposition", statements: v5 },
  { version: 6, name: "d5_blinded_repeat_presentations", statements: v6 },
  { version: 7, name: "d5_campaign_closeout", statements: v7 }
];

export function migrationDigest(migration: Migration): string {
  return createHash("sha256")
    .update(JSON.stringify({ version: migration.version, name: migration.name, statements: migration.statements }))
    .digest("hex");
}

export const requiredTables = [
  "program",
  "category",
  "concept_family",
  "family_version",
  "family_projection",
  "transformation_edge",
  "commitment",
  "dialogue",
  "message",
  "message_version",
  "generation_campaign",
  "generation_task",
  "model_call",
  "candidate",
  "candidate_version",
  "candidate_failure",
  "review",
  "analysis_method",
  "analysis_run",
  "analysis_metric",
  "similarity_edge",
  "template_signature",
  "analysis_run_correction",
  "family_synthesis_assignment",
  "family_synthesis",
  "family_synthesis_basis",
  "structural_disposition",
  "structural_disposition_basis",
  "review_presentation_session",
  "review_presentation",
  "review_presentation_response",
  "review_presentation_score",
  "review_presentation_finding",
  "campaign_closeout_assignment",
  "campaign_closeout",
  "campaign_closeout_state",
  "campaign_closeout_basis",
  "campaign_failure_cluster",
  "campaign_failure_cluster_member",
  "campaign_distribution_assessment",
  "dataset_release",
  "rendered_unit",
  "training_exposure",
  "evaluation_output",
  "event"
] as const;

export const requiredViews = [
  "corpus_candidate_current",
  "public_training_candidate",
  "campaign_progress",
  "candidate_review_state",
  "review_repeat_stability"
] as const;
