import React from "react";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";
import { wipe } from "@remotion/transitions/wipe";

import { IntroScene } from "./scenes/Intro";
import { LossCurveScene } from "./scenes/LossCurve";
import { GpuInfraScene } from "./scenes/GpuInfra";
import { EvolutionScene } from "./scenes/Evolution";
import { LineageTreeScene } from "./scenes/LineageTree";
import { CusumStabilityScene } from "./scenes/CusumStability";
import { ThermoMetricsScene } from "./scenes/ThermoMetrics";
import { RadialVizScene } from "./scenes/RadialViz";
import { ConclusionScene } from "./scenes/Conclusion";

// Scene durations in frames (at 30fps)
// Total target: ~14400 frames (8 minutes)
const INTRO = 420;          // 14s - title, config, symbio explanation
const LOSS_CURVE = 2100;    // 70s - main loss chart draw-on
const GPU_INFRA = 1350;     // 45s - GPU, VRAM, timing
const EVOLUTION = 1800;     // 60s - evolutionary search generations
const LINEAGE_TREE = 2100;  // 70s - tree diagram + force graph
const CUSUM = 1500;         // 50s - CUSUM, clipping, stability
const THERMO = 1500;        // 50s - entropy, free energy, fitness
const RADIAL = 2100;        // 70s - radial polar visualization
const CONCLUSION = 1350;    // 45s - final stats, summary

const TRANSITION_DURATION = 20;

export const SymbioFilm: React.FC = () => {
  return (
    <TransitionSeries>
      {/* 1. Intro - Symbio title and model config */}
      <TransitionSeries.Sequence durationInFrames={INTRO}>
        <IntroScene />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: TRANSITION_DURATION })}
      />

      {/* 2. Loss Curve - Main training loss visualization */}
      <TransitionSeries.Sequence durationInFrames={LOSS_CURVE}>
        <LossCurveScene />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={slide({ direction: "from-left" })}
        timing={linearTiming({ durationInFrames: TRANSITION_DURATION })}
      />

      {/* 3. GPU & Infrastructure */}
      <TransitionSeries.Sequence durationInFrames={GPU_INFRA}>
        <GpuInfraScene />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: TRANSITION_DURATION })}
      />

      {/* 4. Evolutionary Search */}
      <TransitionSeries.Sequence durationInFrames={EVOLUTION}>
        <EvolutionScene />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={wipe({ direction: "from-left" })}
        timing={linearTiming({ durationInFrames: TRANSITION_DURATION })}
      />

      {/* 5. Lineage Tree & Force Graph */}
      <TransitionSeries.Sequence durationInFrames={LINEAGE_TREE}>
        <LineageTreeScene />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: TRANSITION_DURATION })}
      />

      {/* 6. CUSUM & Stability Monitoring */}
      <TransitionSeries.Sequence durationInFrames={CUSUM}>
        <CusumStabilityScene />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: TRANSITION_DURATION })}
      />

      {/* 7. Thermodynamic Metrics */}
      <TransitionSeries.Sequence durationInFrames={THERMO}>
        <ThermoMetricsScene />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={slide({ direction: "from-bottom" })}
        timing={linearTiming({ durationInFrames: TRANSITION_DURATION })}
      />

      {/* 8. Radial Visualization */}
      <TransitionSeries.Sequence durationInFrames={RADIAL}>
        <RadialVizScene />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: TRANSITION_DURATION })}
      />

      {/* 9. Conclusion */}
      <TransitionSeries.Sequence durationInFrames={CONCLUSION}>
        <ConclusionScene />
      </TransitionSeries.Sequence>
    </TransitionSeries>
  );
};
