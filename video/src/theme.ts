/**
 * The video wears the product's own design system, from .stitch/DESIGN.md: paper canvas, one
 * orange accent, ink text, hairline rules, no shadows. A demo film that invents its own look
 * is showing you something other than the thing it is selling.
 */
export const theme = {
  paper: "#F7F7F5",
  panel: "#FFFFFF",
  ink: "#111214",
  graphite: "#5B6068",
  ash: "#8A9099",
  rule: "#E4E4E1",
  accent: "#F4511E",
  clay: "#B4442C",
  sans: '"Geist", -apple-system, "Segoe UI", system-ui, sans-serif',
  mono: '"Geist Mono", ui-monospace, "SF Mono", Menlo, monospace',
} as const;

export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;
