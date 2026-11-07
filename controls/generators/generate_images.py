"""
Batch image generation for the cross-generator check.

Reads prompts_sample.csv (48 corpus prompts verbatim, 8 per subculture, seed 42)
and generates ONE image per prompt per generator. First output is kept; there is
no retry-on-content and no regeneration, so the no-cherry-picking rule is
enforced by construction. Refusals and API errors are logged, not retried.

Generators:
  gpt      OpenAI Images API, model gpt-image-1        (needs OPENAI_API_KEY)
  gpt2     OpenAI Images API, model gpt-image-2        (needs OPENAI_API_KEY)
  gemini   Google Gemini API, gemini-3.1-flash-image   (needs GEMINI_API_KEY)
           (the model publicly nicknamed "Nano Banana 2")
  ideogram Ideogram V4 API, rendering_speed DEFAULT    (needs IDEOGRAM_API_KEY)
           (replaces the Adobe arm: Firefly's API is enterprise-only; vendor
            prompt preprocessing, e.g. magic prompt, counts as part of the
            generator under test, same stance as the other arms)

Keys are read from .env in this folder (KEY=value lines) or the environment.
Images land in images/<generator>/<subculture>/<prompt_id>.png. The script is
resumable: a prompt whose image file already exists is skipped, and a prompt
already logged as refused stays refused (rerunning must not turn a refusal into
a second attempt).

Usage:
  python3 generate_images.py --generators gpt,gemini
  python3 generate_images.py --generators gpt --limit 2   # smoke test
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in _Path(__file__).resolve().parents:
    if (_p / "paths.py").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
from paths import DATASET, CODE_ROOT, OUTPUT, under
for _d in (
    CODE_ROOT / "results" / "laion_profile",
    CODE_ROOT / "results" / "separability",
    CODE_ROOT / "results" / "matched_text",
    CODE_ROOT / "dataset",
    CODE_ROOT / "scoring",
):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))


import argparse
import base64
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = OUTPUT / "generation_log.csv"
LOG_FIELDS = ["generator", "subculture", "prompt_id", "status", "notes"]


def load_env() -> None:
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def post_json(url: str, payload: dict, headers: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gen_openai(prompt: str, model: str) -> tuple[bytes | None, str]:
    """Returns (png_bytes, note). None bytes => refusal/error, note says which."""
    try:
        out = post_json(
            "https://api.openai.com/v1/images/generations",
            {"model": model, "prompt": prompt, "size": "1024x1024",
             "quality": "medium", "n": 1},
            {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        return base64.b64decode(out["data"][0]["b64_json"]), "ok"
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        kind = "refusal" if e.code == 400 and ("moderation" in body or "safety" in body
                                               or "content_policy" in body) else f"http {e.code}"
        return None, f"{kind}: {body}"
    except Exception as e:  # noqa: BLE001 — log and move on, batch must finish
        return None, f"error: {e}"


def gen_gemini(prompt: str) -> tuple[bytes | None, str]:
    try:
        out = post_json(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-3.1-flash-image:generateContent",
            {"contents": [{"parts": [{"text": prompt}]}],
             "generationConfig": {"responseModalities": ["IMAGE"]}},
            {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        cand = out.get("candidates", [{}])[0]
        for part in cand.get("content", {}).get("parts", []):
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"]), "ok"
        reason = cand.get("finishReason", "no image part in response")
        return None, f"refusal: {reason}"
    except urllib.error.HTTPError as e:
        return None, f"http {e.code}: {e.read().decode(errors='replace')[:300]}"
    except Exception as e:  # noqa: BLE001
        return None, f"error: {e}"


def gen_ideogram(prompt: str) -> tuple[bytes | None, str]:
    boundary = "----subcultureboundary7f3a"
    parts = []
    for name, value in (("text_prompt", prompt), ("resolution", "1024x1024"),
                        ("rendering_speed", "DEFAULT")):
        parts += [f"--{boundary}\r\nContent-Disposition: form-data; "
                  f"name=\"{name}\"\r\n\r\n{value}\r\n"]
    body = ("".join(parts) + f"--{boundary}--\r\n").encode()
    try:
        ua = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        req = urllib.request.Request(
            "https://api.ideogram.ai/v1/ideogram-v4/generate", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                     "Api-Key": os.environ["IDEOGRAM_API_KEY"], "User-Agent": ua})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
        data = out.get("data", [])
        if not data or not data[0].get("url"):
            return None, "refusal: empty data array"
        if data[0].get("is_image_safe") is False:
            return None, "refusal: is_image_safe false"
        dl = urllib.request.Request(data[0]["url"], headers={"User-Agent": ua})
        with urllib.request.urlopen(dl, timeout=300) as r:
            return r.read(), "ok"
    except urllib.error.HTTPError as e:
        return None, f"http {e.code}: {e.read().decode(errors='replace')[:300]}"
    except Exception as e:  # noqa: BLE001
        return None, f"error: {e}"


GENERATORS = {"gpt": (lambda p: gen_openai(p, "gpt-image-1"), "OPENAI_API_KEY"),
              "gpt2": (lambda p: gen_openai(p, "gpt-image-2"), "OPENAI_API_KEY"),
              "gemini": (gen_gemini, "GEMINI_API_KEY"),
              "ideogram": (gen_ideogram, "IDEOGRAM_API_KEY")}


def read_log() -> dict[tuple[str, str], dict]:
    done: dict[tuple[str, str], dict] = {}
    if LOG.exists():
        with LOG.open() as f:
            for row in csv.DictReader(f):
                if row.get("prompt_id"):
                    done[(row["generator"], row["prompt_id"])] = row
    return done


def append_log(row: dict) -> None:
    new = not LOG.exists() or LOG.stat().st_size == 0 or "prompt_id" not in LOG.read_text().splitlines()[0]
    if new:  # replace the old placeholder header once
        LOG.write_text(",".join(LOG_FIELDS) + "\n")
    with LOG.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", required=True,
                    help="comma list from: " + ",".join(GENERATORS))
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke test)")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between calls")
    args = ap.parse_args()

    load_env()
    gens = [g.strip() for g in args.generators.split(",")]
    for g in gens:
        if g not in GENERATORS:
            raise SystemExit(f"[FAIL] unknown generator {g!r}")
        if GENERATORS[g][1] not in os.environ:
            raise SystemExit(f"[FAIL] {GENERATORS[g][1]} not set (put it in {HERE / '.env'})")

    prompts = list(csv.DictReader((DATASET / "cross_generator" / "prompts.csv").open()))
    if args.limit:
        prompts = prompts[:args.limit]
    logged = read_log()

    for g in gens:
        fn = GENERATORS[g][0]
        n_ok = n_ref = n_err = n_skip = 0
        for row in prompts:
            pid, sub, prompt = row["prompt_id"], row["subculture"], row["prompt"]
            dest = OUTPUT / "cross_generator" / "images" / g / sub / f"{pid}.png"
            prior = logged.get((g, pid))
            if dest.exists() or (prior and prior["status"] == "refusal"):
                n_skip += 1
                continue
            img, note = fn(prompt)
            status = "ok" if img else ("refusal" if note.startswith("refusal") else "error")
            if img:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(img)
                n_ok += 1
            elif status == "refusal":
                n_ref += 1
            else:
                n_err += 1
            append_log({"generator": g, "subculture": sub, "prompt_id": pid,
                        "status": status, "notes": "" if img else note})
            print(f"[{g}] {pid} {sub}: {status}" + ("" if img else f" ({note[:120]})"), flush=True)
            time.sleep(args.sleep)
        print(f"[{g}] done — ok {n_ok}, refusal {n_ref}, error {n_err}, skipped {n_skip}")
    print("[next] rerun to retry transient errors (refusals stay); then score_generators.py")


if __name__ == "__main__":
    main()
