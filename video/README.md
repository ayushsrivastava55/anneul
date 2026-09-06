# The demo film

Five minutes forty seconds, 1920x1080, narrated. Built with Remotion, voiced by Sarvam.

```
uv run python video/script/tts.py       # narration.json -> public/audio/*.wav
uv run python video/script/timings.py   # measures those files -> timings.json
cd video && npx remotion render src/index.ts Anneal out/anneal-demo.mp4
```

## How it is put together

`script/narration.json` is the whole film: seventeen scenes, each declaring what it is (a
title, a pitch slide, or a screenshot of the running product) and what is said over it. Nothing
about the film's length is written down anywhere. `timings.py` measures the rendered WAV of
each scene and writes `timings.json`; the composition reads that. Re-record a line and the
video re-times itself, so narration and picture cannot drift apart.

Before the key is present there is no audio to measure, so `timings.py` falls back to a
speaking rate and the film renders silent at roughly the right length. Both paths write the
same shape, so the composition never knows which one it got.

## Why the screenshots are screenshots

`public/shots/` is the product actually running, captured through the browser. A demo film that
rebuilds the interface in the film is showing you something other than the thing it is selling.
The frames are dense with real numbers, so the only motion on them is a slow push in; a
screenshot that swings around is a screenshot nobody reads.

The film wears the product's own design system from `.stitch/DESIGN.md`: paper canvas, one
orange accent, ink text, hairline rules, no shadows.

## Sarvam

`SARVAM_API_KEY` in `.env`, which is gitignored. One request per scene against
`POST https://api.sarvam.ai/text-to-speech`, `en-IN`, model `bulbul:v3`. One request per scene
rather than one for the film means a reworded line costs one call, not a whole re-record.
Nothing in `anneal/` reads this key and no run needs it.
