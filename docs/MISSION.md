# Mission: Evolutionary Loss Intelligence

We're building two new lines on the main loss curve chart that understand evolutionary activation search — where the loss jumps around as candidates switch, making traditional "best val" and "overfit" detection useless.

## 1. Evo Best Line (thick green `#10b981`)

A smooth, thick line that tracks the **best achievable loss envelope** through evolutionary chaos. Unlike the existing Best Val diamond (single point at lowest val loss), this is a continuous line that:

- **Snaps down immediately** when any candidate beats the running best
- **Drifts up slowly** (tau ~200 steps) when loss is worse, so dead candidates don't create permanent false floors
- **Smoothed with EMA** to remove jitter from candidate switches
- Drawn as a **thick 3px solid line with glow** — the hero line you watch to see if evolution is actually making progress

The existing Best Val diamond stays. This new line sits alongside it as the evolutionary-aware equivalent.

## 2. Evo Overfit Regions (orange `#f97316` shaded bands)

Instead of a single "overfit" vertical dashed line (which fires once when val loss rises after its global minimum), detect **per-candidate overfit** — shaded vertical bands showing where individual candidates have exhausted their capacity:

- Group metrics by `symbio_candidate_id`
- Find each candidate's local minimum loss
- If loss rises for 5+ consecutive steps after that minimum → that's an overfit region
- Draw as **translucent orange vertical bands** spanning the chart height
- Label with "evo overfit" at the top of each band

This tells you: "this candidate was done learning and we were wasting steps on it." The original overfit line stays for non-evolutionary runs.

## Implementation

All in `apps/web/src/components/charts.tsx`:

1. Add `evoValEnvelope` and `evoOverfit` to `MarkerType` union, defaults, colors, labels
2. Add `evoValEnvelope: {step,loss}[]` and `evoOverfitRegions: {startStep,endStep,candidateId}[]` to `ComputedEvents`
3. Add `computeEvoValEnvelope()` and `computeEvoOverfitRegions()` detection functions
4. Wire into `computeEvents()`
5. Draw the evo best line after the train loss line (thick, glowing, green)
6. Draw evo overfit bands before event markers (translucent orange rectangles)
7. Both toggleable via the existing marker toggle UI
