"""Turn narration.json into one WAV per scene using Sarvam's text-to-speech API.

Written against https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert :
POST https://api.sarvam.ai/text-to-speech, authenticated with an `api-subscription-key`
header, returning `{"request_id": ..., "audios": ["<base64>"]}`. bulbul:v3 accepts 2500
characters per call, which every scene here is comfortably under, and `en-IN` is the Indian
English voice the brief asked for.

Each scene is one call and one file, so a reworded scene costs one request rather than a whole
re-render, and the video's timing can be measured from the audio that actually exists rather
than guessed from a word count.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from urllib import error, request

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "narration.json"
OUT = HERE.parent / "public" / "audio"
ENDPOINT = "https://api.sarvam.ai/text-to-speech"


def api_key() -> str:
    """The key from the environment, or from .env, which is where this repo keeps secrets."""
    key = (os.environ.get("SARVAM_API_KEY") or "").strip()
    if key:
        return key
    env = HERE.parent.parent / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "SARVAM_API_KEY":
                return value.strip().strip("'\"")
    raise SystemExit(
        "SARVAM_API_KEY is not set. Add it to .env as SARVAM_API_KEY=... and run this again."
    )


def speak(text: str, voice: dict, key: str) -> bytes:
    body = json.dumps({"text": text, **voice}).encode("utf-8")
    req = request.Request(
        ENDPOINT,
        data=body,
        headers={"api-subscription-key": key, "Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            payload = json.load(response)
    except error.HTTPError as exc:
        raise SystemExit(f"Sarvam returned {exc.code}: {exc.read().decode('utf-8')[:300]}") from exc
    audios = payload.get("audios") or []
    if not audios:
        raise SystemExit(f"Sarvam returned no audio: {str(payload)[:300]}")
    return base64.b64decode(audios[0])


def seconds(path: Path) -> float:
    """How long a rendered scene actually is. The video is timed from this, not from a guess."""
    with wave.open(str(path)) as handle:
        return handle.getnframes() / float(handle.getframerate())


def main() -> int:
    spec = json.loads(SCRIPT.read_text(encoding="utf-8"))
    voice = spec["voice"]
    key = api_key()
    OUT.mkdir(parents=True, exist_ok=True)
    timings = {}
    for scene in spec["scenes"]:
        path = OUT / f"{scene['id']}.wav"
        if not path.exists() or "--force" in sys.argv:
            print(f"speaking {scene['id']} ...", flush=True)
            path.write_bytes(speak(scene["text"], voice, key))
        timings[scene["id"]] = round(seconds(path), 3)
    (HERE / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    total = sum(timings.values())
    print(f"{len(timings)} scenes, {total:.1f}s total ({total / 60:.1f} min)")
    return 0


def check_ffmpeg() -> None:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
