import React from "react";
import {AbsoluteFill, Audio, Sequence, staticFile} from "remotion";
import {TransitionSeries, linearTiming} from "@remotion/transitions";
import {fade} from "@remotion/transitions/fade";
import {Shot} from "./Shot";
import {Slide} from "./Slide";
import {FPS, theme} from "./theme";
import {Title} from "./Title";
import narration from "./script/narration.json";
import timings from "./script/timings.json";

/**
 * The whole film: seventeen scenes, each held for exactly as long as its narration runs.
 *
 * Scene length comes from timings.json rather than from a number typed here, so re-recording
 * the voice re-times the video and nothing drifts out of sync. Slides and screenshots are the
 * same kind of thing to this file; what differs is only what each scene declares itself to be.
 */

type Scene = (typeof narration.scenes)[number];

const TRANSITION = 14; // frames of cross-fade, subtracted from the scene it overlaps

const frames = (id: string): number => {
  const entry = (timings.scenes as Record<string, {seconds: number}>)[id];
  return Math.max(Math.round((entry?.seconds ?? 4) * FPS), 2 * FPS);
};

export const filmDuration = (): number => {
  const total = narration.scenes.reduce((sum, scene) => sum + frames(scene.id), 0);
  return total - TRANSITION * (narration.scenes.length - 1);
};

const Panel: React.FC<{scene: Scene; index: number; total: number}> = ({
  scene,
  index,
  total,
}) => {
  if (scene.kind === "title") {
    return <Title title={scene.title ?? ""} subtitle={scene.subtitle ?? ""} />;
  }
  if (scene.kind === "slide") {
    return (
      <Slide
        title={scene.title ?? ""}
        bullets={scene.bullets ?? []}
        index={index}
        total={total}
      />
    );
  }
  return (
    <Shot
      src={`shots/${scene.shot}.jpg`}
      caption={scene.caption ?? ""}
      durationInFrames={frames(scene.id)}
    />
  );
};

export const Film: React.FC = () => {
  const slides = narration.scenes.filter((s) => s.kind === "slide").length;
  let slideNumber = 0;
  let elapsed = 0;

  return (
    <AbsoluteFill style={{background: theme.paper}}>
      <TransitionSeries>
        {narration.scenes.map((scene, i) => {
          if (scene.kind === "slide") slideNumber += 1;
          const held = frames(scene.id);
          const element = (
            <React.Fragment key={scene.id}>
              <TransitionSeries.Sequence durationInFrames={held}>
                <Panel scene={scene} index={slideNumber} total={slides} />
              </TransitionSeries.Sequence>
              {i < narration.scenes.length - 1 ? (
                <TransitionSeries.Transition
                  presentation={fade()}
                  timing={linearTiming({durationInFrames: TRANSITION})}
                />
              ) : null}
            </React.Fragment>
          );
          return element;
        })}
      </TransitionSeries>
      {timings.voiced > 0
        ? narration.scenes.map((scene) => {
            const at = elapsed;
            elapsed += frames(scene.id) - TRANSITION;
            return (
              <Sequence key={scene.id} from={at} durationInFrames={frames(scene.id)}>
                <Audio src={staticFile(`audio/${scene.id}.wav`)} />
              </Sequence>
            );
          })
        : null}
    </AbsoluteFill>
  );
};
