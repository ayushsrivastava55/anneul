import React from "react";
import {Composition, staticFile} from "remotion";
import {Film, filmDuration} from "./Film";
import {FPS, HEIGHT, WIDTH} from "./theme";
import narration from "./script/narration.json";
import timings from "./script/timings.json";

export const RemotionRoot: React.FC = () => (
  <Composition
    id="Anneal"
    component={Film}
    durationInFrames={filmDuration()}
    fps={FPS}
    width={WIDTH}
    height={HEIGHT}
  />
);
