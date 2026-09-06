"""Write timings.json: how long each scene runs, and whether real audio backs it.

Until the Sarvam key is present there is no audio to measure, so the film is timed from a
speaking rate. The moment tts.py has run, the same file is rewritten from the length of the
WAV files themselves, and the video re-times to the voice rather than the voice being squeezed
into a guess. Both paths write the same shape, so the composition never knows the difference.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIO = HERE.parent / "public" / "audio"
WORDS_PER_MINUTE = 152  # measured against Sarvam bulbul at pace 0.95
HOLD = 0.9  # seconds of quiet after each scene, so lines do not run into each other


def estimate(text: str) -> float:
    return len(text.split()) / WORDS_PER_MINUTE * 60


def measured(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return handle.getnframes() / float(handle.getframerate())


def main() -> int:
    spec = json.loads((HERE / "narration.json").read_text(encoding="utf-8"))
    scenes = {}
    voiced = 0
    for scene in spec["scenes"]:
        wav = AUDIO / f"{scene['id']}.wav"
        if wav.is_file():
            seconds, source = measured(wav), "audio"
            voiced += 1
        else:
            seconds, source = estimate(scene["text"]), "estimate"
        scenes[scene["id"]] = {"seconds": round(seconds + HOLD, 3), "source": source}
    total = sum(s["seconds"] for s in scenes.values())
    out = {"scenes": scenes, "voiced": voiced, "total_seconds": round(total, 2)}
    for target in (HERE / "timings.json", HERE.parent / "src" / "script" / "timings.json"):
        target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{len(scenes)} scenes, {voiced} voiced, {total / 60:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
