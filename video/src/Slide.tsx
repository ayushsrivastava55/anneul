import React from "react";
import {interpolate, spring, useCurrentFrame, useVideoConfig} from "remotion";
import {theme} from "./theme";

/**
 * A pitch slide: a heading and up to four lines, each arriving after the one before it.
 *
 * The stagger is not decoration. The narration reaches each line at roughly the moment it
 * appears, so the viewer reads what they are hearing instead of reading ahead.
 */
export const Slide: React.FC<{
  title: string;
  bullets: string[];
  index: number;
  total: number;
}> = ({title, bullets, index, total}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const rise = spring({frame, fps, config: {damping: 200}});

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: theme.paper,
        fontFamily: theme.sans,
        padding: "120px 140px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          fontFamily: theme.mono,
          fontSize: 22,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: theme.ash,
          marginBottom: 28,
          opacity: rise,
        }}
      >
        {String(index).padStart(2, "0")} / {String(total).padStart(2, "0")}
      </div>
      <h1
        style={{
          fontSize: 92,
          lineHeight: 1.05,
          letterSpacing: "-0.03em",
          fontWeight: 500,
          color: theme.ink,
          margin: 0,
          transform: `translateY(${(1 - rise) * 18}px)`,
          opacity: rise,
        }}
      >
        {title}
      </h1>
      <div style={{marginTop: 64, display: "flex", flexDirection: "column", gap: 30}}>
        {bullets.map((line, i) => {
          const at = 14 + i * 12;
          const shown = interpolate(frame, [at, at + 16], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={line}
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: 26,
                opacity: shown,
                transform: `translateY(${(1 - shown) * 12}px)`,
              }}
            >
              <span
                style={{
                  width: 34,
                  height: 2,
                  background: i === 0 ? theme.accent : theme.rule,
                  display: "block",
                }}
              />
              <span style={{fontSize: 44, color: theme.ink, letterSpacing: "-0.01em"}}>
                {line}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
