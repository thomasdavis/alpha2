# Operation Lowering Guide

## Example: `alpha.gemm.gemm-bias-gelu`

1. **Alpha** declares tensor semantics, dtype promotion, broadcasting, and gradients.
2. **Helios** chooses a fused or decomposed implementation under shape, precision, memory, and determinism constraints.
3. **Prometheus** expresses contraction, tiled transfers, accumulator type, and an epilogue tree.
4. **Hephaestus** allocates fragments/registers and emits `sm_86` memory, MMA/SIMT, synchronization, and epilogue instructions.
5. **Chronos** supplies ordering tokens and completion semantics.
6. **Hermes** constructs QMD/pushbuffer submission and patches parameters.
7. **Gaia** supplies code, parameter, input, output, and workspace mappings.
8. **Aether** owns every RM/ioctl interaction required to create those objects.

## Example: event-driven mechanism

`alpha.sequence_recurrence.eventDrivenMechanism` lowers to fixed-wave semantics first. Prometheus emits mailbox/state operations; Chronos owns wave and credit rules; Hermes transports event packets; Gaia stores session state and mailboxes. Asynchronous quiescence is a separate implementation choice, not part of the mathematical name.

## Required implementation record

Every non-stub implementation must attach:

- semantic version and operation ID;
- supported dtype/layout/shape region;
- exactness and determinism contract;
- Prometheus lowering or direct-kernel justification;
- generated SASS/resource manifest;
- correctness oracle and tolerance;
- benchmark contract and hardware manifest;
- known failure regions;
- fallback operation ID;
- source or experiment references.
