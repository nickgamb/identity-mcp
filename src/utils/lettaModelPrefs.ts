import fs from "fs";
import path from "path";
import { config } from "../config";

export const LETTA_MODEL_PREFS_FILE = "letta-model-prefs.json";

export interface LettaModelPrefs {
  model: string;
  embedding: string;
  updatedAt: string;
}

export function lettaModelPrefsPath(): string {
  return path.join(config.MEMORY_DIR, LETTA_MODEL_PREFS_FILE);
}

export function readLettaModelPrefs(): LettaModelPrefs | null {
  const prefsPath = lettaModelPrefsPath();
  try {
    if (!fs.existsSync(prefsPath)) return null;
    const raw = JSON.parse(fs.readFileSync(prefsPath, "utf-8")) as LettaModelPrefs;
    if (!raw?.model?.trim() || !raw?.embedding?.trim()) return null;
    return raw;
  } catch {
    return null;
  }
}

export function writeLettaModelPrefs(model: string, embedding: string): string {
  const prefsPath = lettaModelPrefsPath();
  fs.mkdirSync(path.dirname(prefsPath), { recursive: true });
  const data: LettaModelPrefs = {
    model,
    embedding,
    updatedAt: new Date().toISOString(),
  };
  fs.writeFileSync(prefsPath, JSON.stringify(data, null, 2) + "\n", "utf-8");
  return prefsPath;
}
