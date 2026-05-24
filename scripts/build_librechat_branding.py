#!/usr/bin/env python3
"""Build LibreChat branding assets from universal-agent/static/favicon.png."""
from __future__ import annotations

import base64
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("pip install pillow", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(
    r"C:\Users\Nick\Documents\GitHub\mindgardenai-platform\universal-agent\static\favicon.png"
)
if len(sys.argv) > 1:
    SRC = Path(sys.argv[1])
OUT = ROOT / "librechat-config" / "branding"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img = Image.open(SRC).convert("RGBA")
    for size, name in [
        (16, "favicon-16x16.png"),
        (32, "favicon-32x32.png"),
        (180, "apple-touch-icon-180x180.png"),
    ]:
        img.resize((size, size), Image.Resampling.LANCZOS).save(OUT / name)

    b64 = SRC.read_bytes()
    b64s = base64.b64encode(b64).decode("ascii")
    (OUT / "logo.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
        f'<image width="512" height="512" href="data:image/png;base64,{b64s}"/>'
        f"</svg>\n",
        encoding="utf-8",
    )
    print(f"Wrote branding assets to {OUT}")


if __name__ == "__main__":
    main()
