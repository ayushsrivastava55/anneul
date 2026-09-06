import React from "react";
import {Img, interpolate, staticFile, useCurrentFrame} from "remotion";
import {theme} from "./theme";

/**
 * One screenshot of the running product, drifting slowly toward the viewer.
 *
 * The drift scales the framed panel rather than the image inside it. Scaling the image within
 * a clipping border crops whatever grows past the edge, which quietly ate the right-hand
 * column of every console shot: exactly the numbers the narration is talking about.
 *
 * The drift is small on purpose. These frames are dense with real figures, and a screenshot
 * that swings around is a screenshot nobody reads.
 */
const SHOT_WIDTH = 1500;
const SHOT_RATIO = 713 / 1416; // the capture's own aspect, so nothing is squashed

export const Shot: React.FC<{src: string; caption: string; durationInFrames: number}> = ({
  src,
  caption,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const scale = interpolate(frame, [0, durationInFrames], [1, 1.035]);
  const fade = interpolate(frame, [0, 12], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: theme.paper,
        fontFamily: theme.sans,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 40,
      }}
    >
      <div
        style={{
          width: SHOT_WIDTH,
          height: SHOT_WIDTH * SHOT_RATIO,
          border: `1px solid ${theme.rule}`,
          background: theme.panel,
          opacity: fade,
          transform: `scale(${scale})`,
        }}
      >
        <Img
          src={staticFile(src)}
          style={{display: "block", width: "100%", height: "100%", objectFit: "cover"}}
        />
      </div>
      <div style={{fontSize: 30, color: theme.graphite, opacity: fade}}>{caption}</div>
    </div>
  );
};
