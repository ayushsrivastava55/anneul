import React from "react";
import {interpolate, spring, useCurrentFrame, useVideoConfig} from "remotion";
import {theme} from "./theme";

/** The opening and closing card. One word, one line, one rule between them. */
export const Title: React.FC<{title: string; subtitle: string}> = ({title, subtitle}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const rise = spring({frame, fps, config: {damping: 200}});
  const width = interpolate(frame, [10, 40], [0, 320], {
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
      }}
    >
      <div
        style={{
          fontFamily: theme.mono,
          fontSize: 84,
          letterSpacing: "0.34em",
          color: theme.ink,
          paddingLeft: "0.34em",
          opacity: rise,
          transform: `translateY(${(1 - rise) * 14}px)`,
        }}
      >
        {title.toUpperCase()}
      </div>
      <div style={{height: 2, width, background: theme.accent, margin: "48px 0"}} />
      <div
        style={{
          fontSize: 40,
          color: theme.graphite,
          opacity: interpolate(frame, [24, 44], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        {subtitle}
      </div>
    </div>
  );
};
