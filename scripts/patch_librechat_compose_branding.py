#!/usr/bin/env python3
"""Ensure librechat-api service has MindGarden branding volume mounts."""
import pathlib
import re
import sys

COMPOSE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "~/ai/docker-compose.yml").expanduser()

MOUNTS = [
    "      - ./librechat-config/branding/logo.svg:/app/client/dist/assets/logo.svg:ro",
    "      - ./librechat-config/branding/logo.svg:/app/client/public/assets/logo.svg:ro",
    "      - ./librechat-config/branding/favicon-16x16.png:/app/client/dist/assets/favicon-16x16.png:ro",
    "      - ./librechat-config/branding/favicon-16x16.png:/app/client/public/assets/favicon-16x16.png:ro",
    "      - ./librechat-config/branding/favicon-32x32.png:/app/client/dist/assets/favicon-32x32.png:ro",
    "      - ./librechat-config/branding/favicon-32x32.png:/app/client/public/assets/favicon-32x32.png:ro",
    "      - ./librechat-config/branding/apple-touch-icon-180x180.png:/app/client/dist/assets/apple-touch-icon-180x180.png:ro",
    "      - ./librechat-config/branding/apple-touch-icon-180x180.png:/app/client/public/assets/apple-touch-icon-180x180.png:ro",
    "      - ./librechat-config/branding/auth-overrides.css:/app/client/dist/assets/auth-overrides.css:ro",
    "      - ./librechat-config/branding/auth-overrides.css:/app/client/public/assets/auth-overrides.css:ro",
    "      - ./librechat-config/branding/index.html:/app/client/dist/index.html:ro",
]

text = COMPOSE.read_text(encoding="utf-8")
needle = "      - ./librechat-config/librechat.yaml:/app/librechat.yaml:ro\n"
if needle not in text:
    sys.exit(f"Could not find librechat.yaml volume mount in {COMPOSE}")

text = re.sub(r"^\s+- \./librechat-config/branding/.*\n", "", text, flags=re.M)
insert = needle + "\n".join(MOUNTS) + "\n"
text = text.replace(needle, insert, 1)
COMPOSE.write_text(text, encoding="utf-8")
print(f"Patched {COMPOSE}")
