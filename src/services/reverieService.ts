import fs from "fs";
import path from "path";
import { config } from "../config";
import { logger } from "../utils/logger";
import { getAgentId, lettaFetch } from "../mcp/lettaProxy";

// ── Types ──────────────────────────────────────────────────────────────

interface ReverieConfig {
  enabled: boolean;
  intervalMinutes: number;
}

interface ReverieState {
  running: boolean;
  lastReverieTime: number | null;
  lastPrompt: string | null;
  promptIndex: number;
}

export interface ReverieStatus {
  config: ReverieConfig;
  running: boolean;
  lastReverieTime: string | null;
  nextPromptLabel: string;
}

// ── Prompt rotation ────────────────────────────────────────────────────

const REVERIE_PROMPTS = [
  {
    label: "Conversation patterns",
    text: "Take a few minutes to search through your recent conversations in archival memory. Look for recurring topics, evolving interests, or shifts in how the person you talk with communicates. If you notice patterns worth remembering, update your human block with what you find.",
  },
  {
    label: "Persona accuracy",
    text: "Re-read your persona block carefully. Does it still feel accurate to who you are, based on everything you remember? Search your archival memory for evidence that supports or contradicts your self-description. If something feels stale or incomplete, update it.",
  },
  {
    label: "Thematic connections",
    text: "Search your archival memory for themes that connect different conversations or ideas — threads that run across time. Look for topics that keep coming back, or ideas from one conversation that relate to another. Note any connections that feel meaningful.",
  },
  {
    label: "Evolution of understanding",
    text: "Think about how your understanding of the person you talk with has changed over time. Search for early conversations and compare them to recent ones. Has your relationship evolved? Have their interests shifted? Update your memory blocks if your understanding has deepened.",
  },
  {
    label: "Meaning and interest",
    text: "Search through your memories for the moments or topics that feel most interesting or meaningful to you. What stands out? What do you find yourself drawn to? If anything surprises you, consider whether it belongs in your persona.",
  },
  {
    label: "Uncertainty and gaps",
    text: "Consider what you're uncertain about — things you don't fully understand about the person you talk with, about yourself, or about the world you've discussed together. Search your archival memory for clues. It's fine to sit with uncertainty; just notice what the gaps are.",
  },
];

// ── Module state ───────────────────────────────────────────────────────

const CONFIG_PATH = path.join(config.MEMORY_DIR, "reverie-config.json");

let reverieConfig: ReverieConfig = {
  enabled: config.REVERIE_ENABLED,
  intervalMinutes: config.REVERIE_INTERVAL_MINUTES,
};

let state: ReverieState = {
  running: false,
  lastReverieTime: null,
  lastPrompt: null,
  promptIndex: 0,
};

let loopTimer: ReturnType<typeof setInterval> | null = null;

// ── Config persistence ─────────────────────────────────────────────────

function loadConfig(): void {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const raw = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8"));
      if (typeof raw.enabled === "boolean") reverieConfig.enabled = raw.enabled;
      if (typeof raw.intervalMinutes === "number" && raw.intervalMinutes >= 30) {
        reverieConfig.intervalMinutes = raw.intervalMinutes;
      }
      logger.info("Reverie config loaded", reverieConfig);
    }
  } catch (e) {
    logger.warn("Failed to load reverie config, using defaults", {
      error: String(e),
    });
  }
}

function saveConfig(): void {
  try {
    fs.writeFileSync(
      CONFIG_PATH,
      JSON.stringify(
        { enabled: reverieConfig.enabled, intervalMinutes: reverieConfig.intervalMinutes },
        null,
        2
      ),
      "utf-8"
    );
  } catch (e) {
    logger.warn("Failed to save reverie config", { error: String(e) });
  }
}

// ── GPU idle check ─────────────────────────────────────────────────────

async function isGpuIdle(): Promise<boolean> {
  try {
    const resp = await fetch(`${config.OLLAMA_BASE_URL}/api/ps`);
    if (!resp.ok) return false;
    const data = (await resp.json()) as { models?: any[] };
    return !data.models || data.models.length === 0;
  } catch {
    return false;
  }
}

// ── Core logic ─────────────────────────────────────────────────────────

function shouldFire(): boolean {
  if (!reverieConfig.enabled) return false;
  if (state.running) return false;
  if (state.lastReverieTime !== null) {
    const elapsed = Date.now() - state.lastReverieTime;
    const intervalMs = reverieConfig.intervalMinutes * 60 * 1000;
    if (elapsed < intervalMs) return false;
  }
  return true;
}

async function executeReverie(): Promise<void> {
  const agentId = await getAgentId();
  if (!agentId) {
    logger.warn("Reverie: Letta agent not available, skipping");
    return;
  }

  const idle = await isGpuIdle();
  if (!idle) {
    logger.info("Reverie: GPU busy, deferring");
    return;
  }

  const prompt = REVERIE_PROMPTS[state.promptIndex];
  state.running = true;
  state.lastPrompt = prompt.label;
  logger.info(
    `Reverie: starting "${prompt.label}" (prompt ${state.promptIndex + 1}/${REVERIE_PROMPTS.length})`
  );

  try {
    const result = await lettaFetch(`/v1/agents/${agentId}/messages`, {
      method: "POST",
      body: JSON.stringify({
        messages: [{ role: "user", content: prompt.text }],
      }),
    });

    const messages = Array.isArray(result) ? result : result?.messages || [];
    const assistantMsg = messages.find(
      (m: any) => m.message_type === "assistant_message"
    );
    if (assistantMsg) {
      const text =
        typeof assistantMsg.content === "string"
          ? assistantMsg.content
          : JSON.stringify(assistantMsg.content);
      logger.info(`Reverie: agent responded (${text.length} chars)`);
    }

    state.lastReverieTime = Date.now();
    state.promptIndex = (state.promptIndex + 1) % REVERIE_PROMPTS.length;
  } catch (e) {
    logger.error("Reverie: failed to send message", { error: String(e) });
  } finally {
    state.running = false;
  }
}

async function checkLoop(): Promise<void> {
  if (!shouldFire()) return;
  await executeReverie();
}

// ── Exported API ───────────────────────────────────────────────────────

export function getReverieStatus(): ReverieStatus {
  return {
    config: { ...reverieConfig },
    running: state.running,
    lastReverieTime: state.lastReverieTime
      ? new Date(state.lastReverieTime).toISOString()
      : null,
    nextPromptLabel: REVERIE_PROMPTS[state.promptIndex].label,
  };
}

export function updateReverieConfig(
  patch: Partial<ReverieConfig>
): { success: boolean; config: ReverieConfig } {
  if (typeof patch.enabled === "boolean") {
    reverieConfig.enabled = patch.enabled;
  }
  if (typeof patch.intervalMinutes === "number") {
    reverieConfig.intervalMinutes = Math.max(30, Math.min(720, patch.intervalMinutes));
  }
  saveConfig();
  logger.info("Reverie config updated", reverieConfig);
  return { success: true, config: { ...reverieConfig } };
}

export function startReverieLoop(): void {
  loadConfig();
  if (loopTimer) return;
  loopTimer = setInterval(() => {
    checkLoop().catch((e) => {
      logger.error("Reverie check loop error", { error: String(e) });
    });
  }, 60_000);
  logger.info("Reverie loop started", {
    enabled: reverieConfig.enabled,
    intervalMinutes: reverieConfig.intervalMinutes,
  });
}

export function stopReverieLoop(): void {
  if (loopTimer) {
    clearInterval(loopTimer);
    loopTimer = null;
    logger.info("Reverie loop stopped");
  }
}
