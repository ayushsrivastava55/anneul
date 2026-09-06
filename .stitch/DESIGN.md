# Design System: Anneal — Agent Engineering Console

Source of truth for every Stitch screen in this project. Two surfaces share one language:
a **dense operator console** (the loop running) and a **calm onboarding interview** (a
non-developer describing a job). Same tokens, different density.

## 1. Visual Theme & Atmosphere

A metallurgy lab that happens to run software. Anneal takes a rough agent and cools it into a
proven one, so the interface reads as **heat resolving into stability**: work in progress glows
ember, work that survived the gate settles to a cool verified green, work the gate threw out is
marked and kept visible rather than hidden.

Restrained, never decorative. Every pixel is either a measurement or the structure holding a
measurement. Panels are separated by hairlines and negative space, not by floating cards.

- **Density:** 8 on the console (cockpit dense), 3 on the interview (gallery airy)
- **Variance:** 7 — asymmetric split columns, never a symmetric three-up grid
- **Motion:** 6 — spring physics, one perpetual pulse on the single active stage, nothing else loops

## 2. Color Palette & Roles

Absolute neutral base, one accent. Status colours are semantic signals, not decoration, and
appear only on data that has a verdict.

- **Void** (#0A0B0D) — page background. Never pure black.
- **Slate Panel** (#141619) — panel and table fill
- **Raised Panel** (#1A1D21) — hovered row, active input
- **Hairline** (#22262B) — 1px structural borders and dividers, the primary separation device
- **Anneal Ember** (#E2622A) — THE accent. Primary action, the active pipeline stage, in-progress
  heat. Saturation held under 80%. Used once per view, never as a gradient wash.
- **Verified Green** (#2F9E79) — semantic only: promoted, passed, the Pareto front
- **Reject Clay** (#C4544F) — semantic only: rejected, hard-fail
- **Bone** (#E8EAED) — primary text
- **Ash** (#7C838C) — secondary text, axis labels, metadata
- **Iron** (#4A5058) — disabled, future pipeline stages, dominated scatter points

## 3. Typography Rules

- **Display & UI:** `Geist` — tracking-tight, hierarchy by weight and colour, never by size alone
- **Mono:** `Geist Mono` — mandatory for every number, score, p-value, model id, trace id, task id,
  latency and currency figure. Density is above 7, so numbers never render in the sans face.
- **Body (interview only):** `Geist`, relaxed leading, 65 characters maximum per line
- **Banned:** `Inter`, system-font stacks, and all serif faces. This is a dashboard.

## 4. Component Stylings

- **Buttons:** flat fill, no outer glow. Primary is Anneal Ember with Void text. Secondary is a
  hairline outline. Active state translates down 1px. One primary action per view.
- **Panels, not cards:** the console separates regions with a 1px Hairline and a panel header in
  Ash small-caps. No drop shadows, no floating elevation. The interview may use a single
  contained panel because there is only one thing on screen.
- **Rows:** tabular rows carry a 2px left border in the status colour when they have a verdict;
  otherwise no border. Hover raises fill to Raised Panel only.
- **Status chips:** small, uppercase, mono, 1px border in the status colour, transparent fill.
  Never a solid pill, never a glow.
- **Inputs (interview):** label above, generous 44px minimum target, focus ring is a 1px Anneal
  Ember border with no glow. Choice questions render as stacked selectable rows, not radio dots.
- **Loading:** skeleton rows matching the exact table geometry. No circular spinners anywhere.
- **Empty states:** a composed line explaining the next action, e.g. "No runs yet — describe a job
  to generate a domain". Never the word "empty" alone.
- **Charts:** 1px axes in Hairline, Ash mono tick labels, no gridline fill, no legend box border.

## 5. Layout Principles

- CSS Grid, max width 1440px, 24px gutters, 16px internal panel padding
- **Console:** asymmetric split — a wide left column (~62%) for candidates and the live run, a
  narrow right column (~38%) for the ledger and the gate verdict. A full-width pipeline rail sits
  above both, a full-width Pareto strip below.
- **Interview:** single centred column, 560px maximum, one question visible at a time, a thin
  progress rail showing five steps.
- No overlapping elements. No absolute-positioned stacking. Every element owns its zone.
- Below 768px every multi-column layout collapses to one column. Horizontal scroll is a failure,
  except inside a wide data table which scrolls within its own panel.

## 6. Motion & Interaction

- Spring physics, stiffness 100, damping 20. No linear easing.
- Exactly one perpetual loop per view: a slow pulse ring on the currently active pipeline stage.
  Nothing else animates forever.
- Table rows and ledger entries mount in a staggered cascade, ~40ms apart.
- Animate `transform` and `opacity` only.

## 7. Anti-Patterns (Banned)

- No emojis, anywhere
- No `Inter`, no serif faces, no system-font fallback stacks as the design choice
- No pure black `#000000`
- No neon or outer-glow shadows, no purple/blue "AI" glow
- No gradient text on headings
- No three equal columns of cards
- No custom mouse cursors
- No filler text: "Scroll to explore", bouncing chevrons, scroll arrows
- No fake round numbers. Every figure on screen must look like it came from a run: `0.767`,
  `$0.266`, `p 0.750`, `190.9s` — never `99.9%` or `50%`
- No generic placeholder names. Use real artefact names from this system: `cand-01`,
  `cand-01-rewrite_tool_desc-i1`, `invoices`, `airline`, `qwen2.5:3b-instruct`
- No AI copywriting clichés: "Elevate", "Seamless", "Unleash", "Next-Gen", "Supercharge"
- No claiming success in the UI copy. The gate rejects more often than it promotes, and the
  interface must show a rejection as calmly and prominently as a promotion.
