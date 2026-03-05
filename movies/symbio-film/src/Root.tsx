import { Composition } from "remotion";
import { SymbioFilm } from "./SymbioFilm";
import { WIDTH, HEIGHT, FPS } from "./theme";

// 8 minutes at 30fps = 14400 frames
const DURATION = 14400;

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="SymbioFilm"
      component={SymbioFilm}
      durationInFrames={DURATION}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
    />
  );
};
