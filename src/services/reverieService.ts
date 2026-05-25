import fs from "fs";
import path from "path";
import { config } from "../config";
import { logger } from "../utils/logger";
import { getAgentId, lettaFetch } from "../mcp/lettaProxy";
import {
  getActivePrompts,
  getReveriePromptsRaw,
  saveReveriePrompts as savePromptsToFile,
  reloadReveriePrompts,
  formatReverieUserContent,
  type ReveriePrompt,
} from "../utils/reveriePrompts";

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
      if (typeof raw.promptIndex === "number" && raw.promptIndex >= 0) {
        state.promptIndex = raw.promptIndex % getActivePrompts().length;
      }
      logger.info("Reverie config loaded", { ...reverieConfig, promptIndex: state.promptIndex });
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
        {
          enabled: reverieConfig.enabled,
          intervalMinutes: reverieConfig.intervalMinutes,
          promptIndex: state.promptIndex,
        },
        null,
        2
      ),
      "utf-8"
    );
  } catch (e) {
    logger.warn("Failed to save reverie config", { error: String(e) });
  }
}

// ── Core logic ─────────────────────────────────────────────────────────
// No GPU idle check needed — Ollama serializes requests per model, so a
// reverie message sent to Letta simply queues behind any active chat and
// runs when the GPU is free.  state.running prevents reveries from
// stacking on each other.

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

  const prompts = getActivePrompts();
  const prompt = prompts[state.promptIndex % prompts.length];
  state.running = true;
  state.lastPrompt = prompt.label;

  // Always advance to next prompt so restarts/failures don't repeat the same one
  state.promptIndex = (state.promptIndex + 1) % prompts.length;
  saveConfig();

  logger.info(
    `Reverie: starting "${prompt.label}" (next will be ${state.promptIndex + 1}/${prompts.length})`
  );

  try {
    const result = await lettaFetch(`/v1/agents/${agentId}/messages`, {
      method: "POST",
      body: JSON.stringify({
        messages: [
          {
            role: "user",
            content: formatReverieUserContent(prompt.label, prompt.text),
          },
        ],
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
    nextPromptLabel: getActivePrompts()[state.promptIndex % getActivePrompts().length].label,
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

export function getReveriePrompts(): ReveriePrompt[] {
  return getReveriePromptsRaw();
}

export function updateReveriePrompts(
  prompts: ReveriePrompt[]
): { success: boolean; count: number; error?: string } {
  if (!Array.isArray(prompts) || prompts.length === 0) {
    return { success: false, count: 0, error: "Prompts array must not be empty" };
  }
  for (const p of prompts) {
    if (!p.label?.trim() || !p.text?.trim()) {
      return { success: false, count: 0, error: "Each prompt needs a label and text" };
    }
  }
  const cleaned = prompts.map((p) => ({ label: p.label.trim(), text: p.text.trim() }));
  savePromptsToFile(cleaned);
  reloadReveriePrompts();
  if (state.promptIndex >= cleaned.length) {
    state.promptIndex = 0;
    saveConfig();
  }
  logger.info("Reverie prompts updated", { count: cleaned.length });
  return { success: true, count: cleaned.length };
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
