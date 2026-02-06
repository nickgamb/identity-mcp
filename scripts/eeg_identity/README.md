# EEG Identity Assurance Scripts

Scripts for EEG-based identity assurance using an EMOTIV Epoc X headset.

For full documentation, see **[docs/EEG_IDENTITY_ASSURANCE.md](../../docs/EEG_IDENTITY_ASSURANCE.md)**.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Enrollment (synthetic mode for testing)
python enroll_brainwaves.py --synthetic

# Authorization (test against enrolled model)
python authorize_brainwaves.py --synthetic

# With EMOTIV hardware
python enroll_brainwaves.py --serial YOUR_SERIAL_NUMBER
python authorize_brainwaves.py --serial YOUR_SERIAL_NUMBER
```

## Dashboard Mode

When launched from the dashboard, both scripts run with `--dashboard` and emit structured `@@EEG_EVENT:` JSON lines that the dashboard renders as visual modals:

- **Enrollment** — Guided experience with checkerboard, animated breathing circle, word stimuli, action instructions, and live progress bars
- **Authorization** — Spectral overlay chart (enrolled vs live), circular similarity gauge, rolling score timeline, and stop control

The scripts communicate with the React frontend via Server-Sent Events (SSE) streamed through the Node.js backend.

## Files

| File | Purpose |
|------|---------|
| `emotiv_reader.py` | EMOTIV Epoc X device abstraction (HID, crypto, synthetic mode) |
| `enroll_brainwaves.py` | Guided enrollment — captures EEG during neurofeedback tasks, trains reference model |
| `authorize_brainwaves.py` | Live authorization — streams EEG, compares against enrolled model, outputs assurance signal |
| `requirements.txt` | Python dependencies |
