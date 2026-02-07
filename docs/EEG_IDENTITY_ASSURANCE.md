# EEG Identity Assurance

A neurophysiological identity assurance system that uses EEG brainwave patterns from an EMOTIV Epoc X headset to produce a continuous confidence signal. Complements the existing conversation-based identity verification by adding a biological layer.

## Overview

Traditional biometrics verify **what you are** (fingerprint, face). Conversation-based identity verifies **how you communicate**. EEG assurance verifies **how your brain responds** — capturing the unique neurophysiological patterns that emerge during structured neurofeedback tasks.

This is not designed as a standalone credential (like FaceID). It acts as a **continuous assurance signal** — an additional layer of confidence that the person at the keyboard is who they enrolled as. Think of it as the neurological equivalent of the Identity MCP's conversation-based continuous evaluation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    HOW IT WORKS                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ENROLLMENT (one-time setup)                                    │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────────┐                                           │
│   │  Guided Tasks   │  ◄── Baseline, meditation, word focus,    │
│   │  (EEG captured) │      actions (blink, fingers, faces)      │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │ Feature Extract │  ◄── Time, frequency, connectivity        │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │  EEG Centroid   │  ◄── Your neural "fingerprint"            │
│   │  (Reference)    │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   AUTHORIZATION (continuous)                                     │
│   ┌─────────────────┐                                           │
│   │ Live EEG stream │  ◄── Compare against centroid             │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   Assurance Signal: 0.0 – 1.0                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Why Centroid-Based (Not a Classifier)

The EEG identity model follows the exact same design philosophy as the conversation-based `train_identity_model.py`. Both build a **reference centroid** rather than training a multi-class classifier:

1. **No negative examples needed.** You don't need EEG from other people to enroll yourself. The centroid represents *you*, and distance from it represents deviation.
2. **Continuous signal, not binary.** Instead of "yes/no" authentication, you get a similarity score that conveys how much the current signal matches the enrolled identity. This is ideal for an assurance layer.
3. **Composable with other signals.** The assurance score can be combined with conversation-based identity verification, OIDC tokens, and other factors in a policy engine — each contributing a confidence weight.

The flow mirrors `train_identity_model.py` exactly:
- Enrollment captures labeled data and extracts features
- Features are normalized with a fitted scaler
- The centroid (mean of all feature vectors) becomes the reference
- Verification computes cosine similarity between live features and the centroid
- Statistical thresholds (1-sigma, 2-sigma from enrollment distribution) calibrate the confidence mapping

## Hardware: EMOTIV Epoc X

The system is built around the EMOTIV Epoc X consumer EEG headset:

| Specification | Value |
|--------------|-------|
| **EEG Channels** | 14 (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4) |
| **Electrode System** | International 10-20 |
| **Sampling Rate** | 128 Hz |
| **Resolution** | 14-bit per channel |
| **Connection** | USB wireless dongle (auto-detected) |
| **Packet Format** | 64 bytes, AES-ECB encrypted |
| **Decryption** | AES-128 key derived from device serial number |

### Connection Methods

The `emotiv_reader.py` abstraction supports:

- **USB Wireless Dongle (primary)** — Plug in the EMOTIV USB dongle. The reader auto-detects it, derives the AES key from the dongle serial, and handles the EPOC X protocol (XOR preprocessing, 2-byte channel parsing, motion packet filtering). No `--serial` flag needed — it reads the serial from the dongle automatically.
- **Synthetic** — No hardware needed. Generates realistic EEG with configurable "identity seeds" for testing the full pipeline without a headset.

> **Note:** The USB wireless dongle must be connected to the machine running the scripts. When using Docker, the dongle is not accessible from inside the container — run the scripts locally instead (see [Local Development](#local-development) below). Bluetooth HID and direct USB cable are not currently supported due to Windows driver limitations with the EPOC X.

## File Structure

All scripts live in `scripts/eeg_identity/`:

```
scripts/eeg_identity/
├── emotiv_reader.py           # Device abstraction layer
│   ├── EmotivReader class     #   Unified HID + synthetic interface
│   ├── AES crypto             #   Packet decryption from serial number
│   ├── Sensor parsing         #   14-bit channel extraction from packets
│   ├── SyntheticGenerator     #   Realistic fake EEG for testing
│   └── Signal processing      #   Bandpass, notch filters, preprocessing
│
├── enroll_brainwaves.py       # Enrollment script
│   ├── Task definitions       #   Baseline, meditation, word focus, actions
│   ├── Feature extraction     #   Time, frequency, connectivity features
│   ├── Centroid training      #   StandardScaler + mean centroid
│   ├── Dashboard events       #   --dashboard mode emits @@EEG_EVENT SSE lines
│   └── Model saving           #   To models/eeg_identity/
│
├── authorize_brainwaves.py    # Authorization script
│   ├── Model loading          #   Centroid, scaler, config, statistics
│   ├── Live EEG capture       #   Configurable window duration
│   ├── Assurance computation  #   Cosine similarity → calibrated score
│   ├── Live spectral export   #   --dashboard mode sends per-window spectral data
│   └── Continuous mode        #   Periodic checks with rolling average
│
└── requirements.txt           # Python dependencies
```

Model artifacts are saved to `models/eeg_identity/`:

```
models/eeg_identity/
├── config.json                # Model configuration, device info, statistics
├── eeg_centroid.npy           # The identity centroid vector (numpy)
├── spectral_profile.json      # Per-channel frequency band power profiles
├── feature_scaler.json        # StandardScaler mean/scale for normalization
└── enrollment_log.json        # Task-by-task quality and metadata log
```

## Enrollment Process

Enrollment guides the user through a series of structured neurofeedback tasks. Each task is designed to elicit distinct neural patterns that, in combination, form a unique identity signature.

### Task Sequence

| # | Task | Duration | What It Captures |
|---|------|----------|-----------------|
| 1 | **Baseline** | 20s | Resting-state EEG — eyes open, relaxed focus on a fixed point. Captures the individual alpha frequency (IAF), default mode network activity, and baseline spectral distribution. |
| 2 | **Meditation** | 20s | Guided box breathing (4s in, 4s hold, 4s out). Captures alpha/theta enhancement during relaxation — the pattern and degree of alpha increase is highly individual. |
| 3 | **Word Focus** (x5) | 10s each | Focus on TREE, BOOK, RIVER, CLOUD, CHAIR one at a time. Captures event-related desynchronization (ERD) during cognitive engagement — alpha suppression and beta enhancement patterns are person-specific. |
| 4 | **Actions** (x6) | 10s each | Blink, right fingers, left fingers, smile, frown, neutral face. Captures motor cortex lateralization (C3/C4 mu rhythm), facial EMG artifacts (which are actually identity-distinctive), and motor planning patterns. |

Total enrollment time: ~3 minutes (configurable via `--task-duration`, `--word-duration`, `--action-duration`).

### Why These Tasks

These tasks are chosen from established neurofeedback and BCI research:

- **Baseline** provides the reference spectral signature. Every person has a slightly different Individual Alpha Frequency (typically 8-12 Hz) and different relative power distributions across bands.
- **Meditation** amplifies alpha/theta patterns. The degree of alpha enhancement during relaxation varies significantly between individuals and is difficult to consciously fake.
- **Word focus** triggers cognitive processing that shows up as characteristic alpha suppression. The spatial distribution of this suppression across the 14 channels is person-specific.
- **Motor/expression tasks** engage the motor cortex and facial muscles, producing lateralized patterns (right finger movement shows in left motor cortex and vice versa). The exact pattern of mu suppression and beta rebound is individual.

## Feature Extraction

Each recorded EEG segment is converted to a feature vector of ~218 dimensions:

### Time-Domain Features (56 dimensions)
For each of the 14 channels:
- **Mean** — DC offset and drift characteristics
- **Variance** — Signal power and dynamic range
- **Skewness** — Asymmetry of the amplitude distribution
- **Kurtosis** — Peakedness, sensitive to artifacts and burst patterns

### Frequency-Domain Features (70 dimensions)
For each of the 14 channels, relative power in 5 standard bands:
- **Delta** (0.5–4 Hz) — Deep processing, often elevated during drowsiness
- **Theta** (4–8 Hz) — Memory encoding, meditation, creative states
- **Alpha** (8–13 Hz) — Relaxed attention, inhibition; most individually variable
- **Beta** (13–30 Hz) — Active cognition, motor planning, focus
- **Gamma** (30–64 Hz) — Higher cognition, binding, perceptual processing

Using *relative* power (band power / total power) makes features robust to amplitude differences between sessions and electrode contact variations.

### Connectivity Features (91 dimensions)
Upper triangle of the 14x14 inter-channel correlation matrix. This captures how brain regions communicate — the functional connectivity pattern is one of the most individually distinctive EEG features.

### Asymmetry Features (1 dimension)
Frontal alpha asymmetry (log F4 alpha – log F3 alpha). This reflects emotional processing lateralization and is a stable individual trait in the neuroscience literature.

## Signal Processing Pipeline

Raw EEG goes through standard preprocessing before feature extraction:

1. **Bandpass filter** (0.5–45 Hz, 4th-order Butterworth) — Removes DC drift and high-frequency noise while keeping all neurologically relevant frequencies.
2. **50 Hz notch filter** — Removes European power line interference.
3. **60 Hz notch filter** — Removes US power line interference.
4. **Quality check** — Rejects segments with amplitude > 500 µV (artifact), variance < 0.01 (flat/disconnected channel), or NaN values.

## Authorization & Assurance Signal

Once enrolled, `authorize_brainwaves.py` captures a window of live EEG and compares it against the enrolled model:

1. Capture N seconds of live EEG (default: 10s)
2. Preprocess (same pipeline as enrollment)
3. Extract features (same function as enrollment)
4. Normalize using the saved scaler parameters
5. Compute cosine similarity to the enrolled centroid
6. Map similarity to a calibrated 0.0–1.0 assurance score using enrollment statistics

### Assurance Score Mapping

| Score Range | Confidence | Meaning |
|------------|------------|---------|
| 0.8–1.0 | **Very High** | Patterns closely match enrollment. Similarity at or above the enrollment mean. |
| 0.6–0.8 | **High** | Strong match. Similarity within 1 standard deviation of enrollment mean. |
| 0.3–0.6 | **Moderate** | Partial match. Similarity within 2 standard deviations. Could indicate the same person in a different cognitive state. |
| 0.0–0.3 | **Low** | Patterns do not match. Similarity below 2 standard deviations from mean. |

### Continuous Mode

With `--continuous`, the script runs indefinitely, producing an assurance signal at regular intervals. It also computes a rolling average of the last 5 checks for smoother signaling:

```bash
python authorize_brainwaves.py --synthetic --continuous --interval 10
```

This is the shape that MCP tools will use during a live chat session — periodically sampling EEG and feeding the assurance score into the identity evaluation alongside conversation-based signals.

## Usage

### Prerequisites

```bash
cd scripts/eeg_identity
pip install -r requirements.txt
```

### Enrollment

```bash
# With EMOTIV hardware (dongle auto-detected)
python enroll_brainwaves.py

# Synthetic mode for testing the full pipeline
python enroll_brainwaves.py --synthetic

# Research edition device with shorter tasks
python enroll_brainwaves.py --device-type research --task-duration 15

# Custom durations
python enroll_brainwaves.py --synthetic --task-duration 10 --word-duration 5 --action-duration 5
```

### Authorization

```bash
# Single check with hardware (dongle auto-detected)
python authorize_brainwaves.py

# Synthetic mode
python authorize_brainwaves.py --synthetic

# Continuous monitoring
python authorize_brainwaves.py --continuous --interval 10

# JSON output for script/tool integration
python authorize_brainwaves.py --synthetic --json

# Simulate a DIFFERENT person (to test rejection)
python authorize_brainwaves.py --synthetic --different-person
```

### Local Development

The EEG scripts need direct access to the USB dongle, which is not available inside Docker containers on Windows. To run with hardware:

```bash
# Terminal 1: Start the MCP server locally
npm run build
$env:PROJECT_ROOT = "."
$env:MEMORY_DIR = "./memory"
$env:FILES_DIR = "./files"
$env:PORT = "4000"
node dist/index.js

# Terminal 2: Start the dashboard
cd dashboard
npm run dev    # http://localhost:3001
```

Ensure `dashboard/vite.config.ts` has the proxy target set to `http://localhost:4000` (not the Docker service name).

The server auto-detects the correct Python command (`python` on Windows, `python3` on Linux/macOS).

### Via Dashboard

Both scripts appear as pipeline cards in the dashboard. When launched from the UI, they open **dedicated visual modals** instead of the standard terminal output panel:

**Enrollment Modal:**
- Full-screen guided experience with dark background for high contrast
- Visual stimuli displayed during each task:
  - **Checkerboard** (SVG grid) for baseline/resting focus
  - **Breathing Circle** (animated SVG with expand-hold-shrink cycle) for meditation
  - **Word Display** (large centered text with fade transitions) for word focus
  - **Instruction Text** for motor/expression action tasks
- Live progress bar with countdown timer during every recording phase
- Task progress tracker showing N of M tasks completed
- Completion screen with summary stats (tasks passed, feature dimensions, similarity metrics) and per-task pass/fail badges

**Authorization Modal:**
- Spectral overlay chart showing enrolled profile (emerald) vs live EEG (cyan) for each channel
- Circular similarity gauge (0.0-1.0) with color transitions (red -> amber -> green)
- Rolling score timeline chart with 1-sigma threshold reference line
- Live reading progress indicator
- Stop button to terminate the session

Both modals communicate with the running Python scripts via **output polling**: the scripts emit structured `@@EEG_EVENT:` JSON lines to stdout, the Node.js backend buffers them, and the React components poll `GET /api/mcp/pipeline.output/:scriptId?cursor=N` every 100ms to receive new lines and render the experience based on the event type.

> **Why polling instead of SSE?** The Vite development proxy buffers and caches `text/event-stream` responses, preventing real-time delivery. JSON polling through the same proxy works reliably with ~100ms latency.

The scripts run in `--dashboard` mode automatically when launched from the UI. Terminal mode (no flag) remains unchanged for CLI use.

### Via MCP Tools

Four tools are registered:

| Tool | Description |
|------|-------------|
| `eeg_model_status` | Check if the EEG identity model is enrolled and available |
| `eeg_enroll` | Run enrollment (supports `mode`, `serial`, `task_duration` parameters) |
| `eeg_authorize` | Run authorization and get assurance signal |
| `eeg_profile_summary` | Inspect the enrolled EEG profile (spectral data, enrollment tasks, statistics) |

### Via HTTP API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mcp/eeg.model_status` | GET | Check model availability |
| `/api/mcp/eeg.enroll` | POST | Run enrollment |
| `/api/mcp/eeg.authorize` | POST | Run authorization |
| `/api/mcp/eeg.profile_summary` | GET | Get profile summary |
| `/api/mcp/pipeline.output/:id?cursor=N` | GET | Poll buffered script output from cursor position (used by dashboard modals) |
| `/api/mcp/pipeline.stream/:id` | GET | SSE stream of script output (alternative, may be buffered by proxies) |
| `/api/mcp/pipeline.stop` | POST | Terminate a running pipeline script |

## Dashboard Visual Architecture

When the scripts run from the dashboard, they use a real-time output polling architecture:

```
┌──────────────┐     ┌───────────────┐     ┌─────────────────┐
│  React Modal │◄────│  Node.js API  │◄────│  Python Script  │
│              │     │  /pipeline    │     │                 │
│  Renders:    │     │  .output/:id  │     │  Emits:         │
│  • Visuals   │ poll│               │     │  @@EEG_EVENT:   │
│  • Charts    │@100ms  Buffers stdout  ←  │  JSON lines     │
│  • Gauges    │     │  lines array  │     │                 │
└──────────────┘     └───────────────┘     └─────────────────┘
```

**Event flow:**

1. Dashboard opens the modal and POSTs to `/api/mcp/pipeline.run` with `--dashboard` flag
2. Modal polls `GET /api/mcp/pipeline.output/:scriptId?cursor=N` every 100ms
3. Python script emits events like `@@EEG_EVENT:{"event":"task_start","visual":"checkerboard",...}`
4. Backend buffers stdout lines; polling endpoint returns new lines since the cursor position
5. React modal finds `@@EEG_EVENT:` markers in each line and renders the corresponding visual component
6. When `done: true` is returned, the modal shows the completion screen

**Key events (enrollment):** `enrollment_start`, `task_start`, `recording_progress`, `task_complete`, `enrollment_complete`

**Key events (authorization):** `auth_start` (includes enrolled spectral profile), `reading_start`, `reading_progress`, `assurance_result` (includes live spectral data for overlay), `auth_stopped`

The `--dashboard` flag is additive — the scripts still print normal terminal output alongside the events. Without the flag, no events are emitted and the scripts behave as before.

## Integration with Conversation-Based Identity

The EEG assurance signal is designed to be composed with the existing conversation-based verification:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Live Chat       │     │  Identity MCP    │     │  Policy Engine   │
│  Session         │────▶│                  │────▶│  (future)        │
│                  │     │  Conversation:   │     │                  │
│  User types      │     │    0.85 (high)   │     │  Combined signal │
│  EEG streams     │     │  EEG Assurance:  │     │  for auth        │
│                  │     │    0.78 (high)   │     │  decisions       │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

The two signals are independent and complementary:
- **Conversation identity** detects if the *writing patterns* match
- **EEG assurance** detects if the *neural patterns* match

A sophisticated impersonator might mimic writing style, but reproducing someone's brainwave patterns during the same tasks is neurophysiologically implausible.

## Privacy & Security

- **All processing is local** — no EEG data leaves your machine
- **You own your model** — it's just numpy/JSON files on disk
- **No cloud dependencies** — works completely offline
- **AES decryption keys** are derived from the device serial number and never stored
- **Raw EEG is not persisted** — only extracted feature vectors and the centroid are saved

## Limitations

1. **Hardware required** — Enrollment and authorization need an EMOTIV Epoc X (or synthetic mode for testing)
2. **Session variability** — EEG can vary with fatigue, caffeine, time of day. Multiple enrollment sessions recommended.
3. **Not credential-grade (yet)** — Designed as an assurance signal, not a standalone authentication factor. Best used alongside other identity signals.
4. **Electrode contact** — Poor contact on some channels degrades accuracy. Quality checks flag issues but can't fix them.
5. **Single-user model** — Like the conversation identity model, each enrolled model represents one person.

## Troubleshooting

### "No EMOTIV devices found"
- Ensure the USB wireless dongle is plugged in
- Ensure the headset is powered on (solid green LED) and paired with the dongle
- On Windows, make sure you're running the scripts locally (not inside Docker — containers can't access USB HID devices)
- On Linux, check udev rules for HID permissions

### Low assurance scores for the enrolled user
- Re-enroll multiple times and across different sessions
- Ensure good electrode contact (saline solution on felt pads)
- Minimize muscle artifacts (jaw clenching, eye movements)
- Use longer capture windows (`--window 20`)

### "No EEG identity model found"
Run enrollment first: `python enroll_brainwaves.py --synthetic`

### Testing without hardware
Use `--synthetic` flag on both scripts. Use `--different-person` on authorize to verify that a different synthetic identity gets a lower score.

## Related Documentation

- **[Identity Verification](./IDENTITY_VERIFICATION.md)** — Conversation-based behavioral identity verification
- **[Getting Started](./GETTING_STARTED.md)** — End-to-end setup with your ChatGPT data
- **[MCP Protocol Reference](./MCP_README.md)** — Complete API reference for all MCP tools
- **[Multi-User & OIDC Support](./MULTI_USER_OIDC.md)** — Multi-user data isolation and OIDC authentication
