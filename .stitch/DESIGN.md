# Design System: Anneal

Source of truth for every Anneal surface: the landing page, the onboarding interview, and the
run console. Calibrated against the approved reference: light, calm, enormously spacious, one
orange accent, all technical content in monospace.

**The governing principle: the interface is a flow, not a cockpit.** A first-time user should be
able to operate it without a legend. When density and clarity conflict, clarity wins and the
detail moves one level deeper.

## 1. Visual Theme & Atmosphere

Swiss editorial calm applied to a technical product. Near-white paper, near-black ink, a single
warm orange used perhaps four times per screen. The product's own output — the goal, the tools,
the scorer, the run trace — is rendered in monospace inside a bordered panel, so the machine's
voice is visually distinct from ours.

- **Density:** 3 — gallery airy. Whitespace is the primary layout tool.
- **Variance:** 6 — asymmetric split (prose left, live panel right). Never centred, never a
  symmetric three-up grid.
- **Motion:** 4 — restrained. Content settles in; nothing bounces, nothing loops except one small
  status dot on the actively running step.

## 2. Color Palette & Roles

- **Paper** (#F7F7F5) — page background, a warm off-white. Never pure white for the canvas.
- **Panel** (#FFFFFF) — the bordered console/product panel fill, so it lifts off Paper by tone
  alone rather than by shadow
- **Ink** (#111214) — headlines and primary text. Never pure black.
- **Graphite** (#5B6068) — body copy, descriptions
- **Ash** (#8A9099) — labels, axis ticks, metadata, timestamps
- **Rule** (#E4E4E1) — 1px borders, dividers, chart gridlines
- **Anneal Orange** (#F4511E) — the ONLY accent. Reserved for: the full stop after the headline,
  the currently running step's dot, the score curve, and the single primary button fill. If it
  appears more than about four times on a screen, something is wrong.
- **Signal Green** (#2F9E79) — semantic only, and used sparingly: a promoted result, a healthy
  status dot
- **Signal Clay** (#C4544F) — semantic only: a rejected mutation, a hard failure

No second accent. No purple, no neon, no gradient fills.

## 3. Typography Rules

- **Display:** `Geist` at 600–700 weight, tracking tight (-0.03em), scaled with `clamp()`. The
  hero headline is genuinely large (about 92px desktop) and sits on 3 short lines. Hierarchy comes
  from weight and colour, not from shouting.
- **Body:** `Geist` 400, relaxed leading, maximum 65 characters per line, in Graphite.
- **Mono:** `Geist Mono` — mandatory for every goal/tool/scorer value, every number, score,
  p-value, timestamp, model id, task id, and every line of the run trace. Section labels above
  panels are mono, uppercase, letterspaced, 11px, in Ash.
- **Banned:** `Inter`, system-font stacks as the design choice, and all serif faces.

## 4. Component Stylings

- **The product panel** (the console on the right of the hero, and the run console page): white
  fill, 1px Rule border, generous internal padding, a thin header strip with a mono path label on
  the left and a version plus status dot on the right. This is the one place elevation exists,
  and it is achieved with a border, not a shadow.
- **Definition rows:** a mono label in Ash on the left (GOAL, TOOLS, SCORER), value in Ink on the
  right. Aligned in a two-column grid. This is how the three inputs are always shown.
- **Step timeline:** a vertical list of steps, each with a small dot on a 1px connector line, a
  mono step name in Ink, a one-line description in Ash, an elapsed timestamp on the right, and a
  chevron for detail. The active step's dot is Anneal Orange; completed dots are Ash; future dots
  are Rule. This is the primary way progress is communicated anywhere in the product.
- **Charts:** 1px Rule gridlines, Ash mono ticks, the measured series as a 2px Anneal Orange line
  with small filled dots, projected or not-yet-run points as a dotted Ash line with hollow dots.
  No filled areas, no legend boxes.
- **Buttons:** one primary per view — Ink fill, Paper text, small right arrow, 8px radius, no
  shadow, no glow. Secondary is a Rule outline on Panel fill. Active state translates down 1px.
- **The explainer row:** four columns separated by 1px vertical rules, each a mono uppercase label
  and two short lines of Graphite prose. This is the exception to the no-equal-columns rule
  because it is a definition list, not a feature grid.
- **Inputs (interview):** one question on screen at a time. Label above in mono uppercase Ash,
  the control below at a minimum 44px target. Choice questions are stacked selectable rows with a
  1px Rule border, not radio dots; the selected row takes a 1px Ink border.
- **Loading:** skeleton lines matching the real geometry. Never a spinner.
- **Empty states:** one sentence naming the next action, e.g. "No runs yet — describe a job to
  generate a domain."

## 5. Layout Principles

- 1440px max width, centred, with large outer margins. 24px gutter, 12-column grid.
- **Hero / landing:** asymmetric split, roughly 46% prose and 54% product panel, vertically
  centred against each other. A small mono eyebrow ("01 — FROM INTENT TO AGENT") sits above the
  headline. Below the fold line, the four-column explainer row. A thin footer strip carries the
  loop as breadcrumbs: BUILD > EVALUATE > IMPROVE > REPEAT.
- **Interview:** a single centred column, 560px maximum. One question visible. A five-step
  progress rail at the top. Back is always available; nothing is destructive.
- **Console:** the same step timeline as the hero panel, expanded. Prose narration on the left,
  the current artefact on the right. Detail (per-task tables, the full ledger) lives behind a
  step's chevron rather than all on one screen. **Do not render every panel at once.**
- No overlapping elements. No absolute-positioned stacking.
- Below 768px everything collapses to one column, in reading order, with no horizontal scroll.

## 6. Motion & Interaction

- Spring physics, stiffness 100, damping 20. No linear easing.
- Exactly one perpetual loop in the entire product: the pulse on the active step's dot.
- Step rows and table rows mount in a staggered cascade about 40ms apart.
- The score curve draws left to right once when it enters the viewport, then rests.
- Animate `transform` and `opacity` only.

## 7. Anti-Patterns (Banned)

- No emojis, anywhere
- No `Inter`, no serif faces
- No pure black `#000000` and no pure white page canvas
- No dark mode for the landing or the interview; the console may invert only if it stays on this
  palette
- No neon or outer-glow shadows, no drop shadows at all — separation is by 1px Rule and tone
- No gradient text, no gradient fills
- No second accent colour
- No custom mouse cursors
- No filler text: "Scroll to explore", bouncing chevrons, scroll arrows
- No AI copywriting clichés: "Elevate", "Seamless", "Unleash", "Next-Gen", "Supercharge",
  "Revolutionise". The reference voice is flat and declarative: "You define the objective."
- No fake round numbers. Every figure shown must look like it came from a run: `0.767`, `$0.266`,
  `p 0.750`, `190.9s`, `iteration 4 / 12` — never `99.9%` or `50%`
- No generic placeholder names. Use real artefacts: `cand-01`,
  `cand-01-rewrite_tool_desc-i1`, `invoices`, `airline`, `qwen2.5:3b-instruct`
- **No cockpit.** If a screen needs a legend to be understood, it has failed. Push detail one
  level deeper instead of adding another panel.
- No claiming success in UI copy. The gate rejects more often than it promotes; a rejection is
  displayed as calmly and as prominently as a promotion.
