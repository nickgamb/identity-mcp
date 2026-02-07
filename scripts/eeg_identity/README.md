# EEG Identity Assurance Scripts

Scripts for EEG-based identity assurance using an EMOTIV Epoc X headset via USB wireless dongle.

For full documentation, see **[docs/EEG_IDENTITY_ASSURANCE.md](../../docs/EEG_IDENTITY_ASSURANCE.md)**.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Enrollment (synthetic mode for testing)
python enroll_brainwaves.py --synthetic

# Authorization (test against enrolled model)
python authorize_brainwaves.py --synthetic

# With EMOTIV hardware (dongle auto-detected, no --serial needed)
python enroll_brainwaves.py
python authorize_brainwaves.py --continuous
```

## Dashboard Mode

When launched from the dashboard, both scripts run with `--dashboard` and emit structured `@@EEG_EVENT:` JSON lines that the dashboard renders as interactive visual modals:

- **Enrollment** — Guided experience with checkerboard, animated breathing circle, word stimuli, action instructions, and smoothly animated progress bars
- **Authorization** — Spectral overlay chart (enrolled vs live), circular similarity gauge, rolling score timeline, and stop control

The React frontend polls the backend's `/api/mcp/pipeline.output/:scriptId` endpoint at 100ms intervals to receive script output in real time. Events can appear anywhere within a stdout line (e.g., after progress bar characters).

> **Important:** The EEG scripts need direct USB access to the EMOTIV dongle. When testing with hardware, run the MCP server and dashboard locally — not inside Docker (containers can't access host USB HID devices on Windows).

## Files

| File | Purpose |
|------|---------|
| `emotiv_reader.py` | EMOTIV Epoc X device abstraction (USB dongle HID, AES-ECB crypto, EPOC X protocol, synthetic mode) |
| `enroll_brainwaves.py` | Guided enrollment — captures EEG during neurofeedback tasks, trains reference centroid model |
| `authorize_brainwaves.py` | Live authorization — streams EEG, compares against enrolled model, outputs assurance signal |
| `requirements.txt` | Python dependencies (hidapi, pycryptodome, numpy, scipy) |
