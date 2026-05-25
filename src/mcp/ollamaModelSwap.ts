/**
 * Ollama VRAM swap when Letta agent model/embedding handles change.
 * Unloads prior weights and warm-loads the new selection (keep_alive: -1).
 */

import { config } from "../config";
import { logger } from "../utils/logger";

const WARM_LOAD_TIMEOUT_MS =
  Number(process.env.OLLAMA_WARM_LOAD_TIMEOUT_MS) || 20 * 60 * 1000;
const UNLOAD_TIMEOUT_MS = 120_000;
const POLL_INTERVAL_MS = 2_000;
const POLL_UNLOAD_MAX_MS = 90_000;
const POLL_LOAD_MAX_MS = WARM_LOAD_TIMEOUT_MS;

export function ollamaNameFromHandle(handle: string): string {
  const trimmed = handle.trim();
  return trimmed.replace(/^ollama\//i, "");
}

function normalizeModelName(name: string): string {
  return name.replace(/:latest$/i, "").toLowerCase();
}

export function modelsMatch(a: string, b: string): boolean {
  const na = normalizeModelName(a);
  const nb = normalizeModelName(b);
  if (!na || !nb) return false;
  return na === nb || na.startsWith(`${nb}:`) || nb.startsWith(`${na}:`);
}

type PsModel = { name: string };

async function listRunningModels(): Promise<string[]> {
  const resp = await fetch(`${config.OLLAMA_BASE_URL}/api/ps`, {
    signal: AbortSignal.timeout(30_000),
  });
  if (!resp.ok) {
    throw new Error(`Ollama ps ${resp.status}`);
  }
  const data = (await resp.json()) as { models?: PsModel[] };
  return (data.models ?? []).map((m) => m.name).filter(Boolean);
}

async function ollamaGenerate(
  body: Record<string, unknown>,
  timeoutMs: number
): Promise<void> {
  const resp = await fetch(`${config.OLLAMA_BASE_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`Ollama generate ${resp.status}: ${text.slice(0, 300)}`);
  }
  // Drain body so the connection closes cleanly (load can take many minutes).
  await resp.text();
}

async function unloadModel(modelName: string): Promise<void> {
  logger.info("Ollama: unloading model", { model: modelName });
  await ollamaGenerate({ model: modelName, keep_alive: 0 }, UNLOAD_TIMEOUT_MS);
}

async function warmModel(modelName: string): Promise<void> {
  logger.info("Ollama: warm-loading model", { model: modelName });
  await ollamaGenerate(
    {
      model: modelName,
      prompt: " ",
      stream: false,
      keep_alive: -1,
      options: { num_predict: 1 },
    },
    WARM_LOAD_TIMEOUT_MS
  );
}

async function waitUntil(
  predicate: () => Promise<boolean>,
  maxMs: number,
  label: string
): Promise<boolean> {
  const deadline = Date.now() + maxMs;
  while (Date.now() < deadline) {
    if (await predicate()) return true;
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
  logger.warn(`Ollama: timed out waiting for ${label}`);
  return false;
}

async function unloadAllExcept(keepName?: string): Promise<string[]> {
  const running = await listRunningModels();
  const unloaded: string[] = [];
  for (const name of running) {
    if (keepName && modelsMatch(name, keepName)) continue;
    try {
      await unloadModel(name);
      unloaded.push(name);
    } catch (e) {
      logger.warn("Ollama: unload failed", { model: name, error: String(e) });
    }
  }
  if (unloaded.length > 0) {
    await waitUntil(async () => {
      const still = await listRunningModels();
      return !still.some((n) => unloaded.some((u) => modelsMatch(n, u)));
    }, POLL_UNLOAD_MAX_MS, "unload");
  }
  return unloaded;
}

async function ensureModelLoaded(modelName: string): Promise<void> {
  const running = await listRunningModels();
  if (running.some((n) => modelsMatch(n, modelName))) {
    logger.info("Ollama: model already loaded", { model: modelName });
    return;
  }
  await warmModel(modelName);
  const loaded = await waitUntil(async () => {
    const now = await listRunningModels();
    return now.some((n) => modelsMatch(n, modelName));
  }, POLL_LOAD_MAX_MS, "load");
  if (!loaded) {
    throw new Error(`Timed out waiting for ${modelName} to appear in ollama ps`);
  }
}

export type OllamaSwapResult = {
  status: "skipped" | "completed" | "failed";
  unloaded: string[];
  loaded?: string;
  embeddingLoaded?: string;
  message?: string;
  error?: string;
};

/**
 * After Letta agent PATCH: unload stale VRAM and pin the new model/embedding.
 */
export async function swapOllamaAfterModelChange(opts: {
  modelPatched?: boolean;
  embeddingPatched?: boolean;
  previousModelHandle?: string;
  newModelHandle?: string;
  previousEmbeddingHandle?: string;
  newEmbeddingHandle?: string;
}): Promise<OllamaSwapResult> {
  const prevModel = opts.previousModelHandle
    ? ollamaNameFromHandle(opts.previousModelHandle)
    : "";
  const newModel = opts.newModelHandle
    ? ollamaNameFromHandle(opts.newModelHandle)
    : "";
  const prevEmbed = opts.previousEmbeddingHandle
    ? ollamaNameFromHandle(opts.previousEmbeddingHandle)
    : "";
  const newEmbed = opts.newEmbeddingHandle
    ? ollamaNameFromHandle(opts.newEmbeddingHandle)
    : "";

  const modelChanged =
    !!newModel && !!prevModel && !modelsMatch(prevModel, newModel);
  const embedChanged =
    !!newEmbed && !!prevEmbed && !modelsMatch(prevEmbed, newEmbed);
  const modelSet = opts.modelPatched === true && !!newModel;
  const embedSet = opts.embeddingPatched === true && !!newEmbed;

  if (!modelSet && !embedSet) {
    return { status: "skipped", unloaded: [], message: "No Ollama swap needed" };
  }

  const unloaded: string[] = [];

  try {
    if (modelSet) {
      if (modelChanged) {
        unloaded.push(...(await unloadAllExcept(undefined)));
      }
      await ensureModelLoaded(newModel);
    }

    if (embedSet && !modelsMatch(newEmbed, newModel)) {
      if (embedChanged && prevEmbed) {
        const running = await listRunningModels();
        if (running.some((n) => modelsMatch(n, prevEmbed))) {
          try {
            await unloadModel(prevEmbed);
            unloaded.push(prevEmbed);
          } catch (e) {
            logger.warn("Ollama: embedding unload failed", {
              model: prevEmbed,
              error: String(e),
            });
          }
        }
      }
      await warmModel(newEmbed);
      return {
        status: "completed",
        unloaded,
        loaded: modelSet ? newModel : undefined,
        embeddingLoaded: newEmbed,
        message: modelSet
          ? `Loaded ${newModel}${embedSet ? `; warmed ${newEmbed}` : ""}`
          : `Warmed embedding ${newEmbed}`,
      };
    }

    return {
      status: "completed",
      unloaded,
      loaded: modelSet ? newModel : undefined,
      message: modelSet ? `Loaded ${newModel}` : undefined,
    };
  } catch (e) {
    const err = String(e);
    logger.error("Ollama model swap failed", { error: err });
    return {
      status: "failed",
      unloaded,
      loaded: modelSet ? newModel : undefined,
      error: err,
    };
  }
}
