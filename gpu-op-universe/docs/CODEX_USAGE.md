# Instructions for Codex and Other Coding Agents

1. Search `catalog/operation-registry.json` before inventing a new API name.
2. Treat generated functions as discoverability stubs, not evidence that an operation exists.
3. Never implement a `research` or `speculative` operation without a short experiment document containing equations, expected savings, baseline, kill criterion, and validation plan.
4. Preserve strict downward dependencies: Alpha → Helios → Prometheus → Hephaestus → Chronos → Hermes → Gaia → Aether.
5. Do not bypass a layer because a lower-level function appears convenient.
6. Prefer parameterized semantics over adding dtype/layout-specific names unless the algorithm or hardware path is genuinely different.
7. Add tests before replacing `defineStub`:
   - reference semantics;
   - edge cases and NaN/Inf behavior;
   - gradient test where applicable;
   - deterministic-mode test;
   - resource/spill test;
   - matched end-to-end benchmark.
8. Record negative results. A stub may remain intentionally unimplemented when a benchmark closes the idea.
9. For `sm_86`, reject unsupported native instructions at compile time; future operations remain legal in the registry but require emulation or another target.
10. Never derive ioctl structures, SASS encodings, or safety-critical control fields from an operation name alone. Use the pinned compatibility profile and verified reverse-engineering evidence.
