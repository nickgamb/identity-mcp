/**
 * EEG Identity Assurance Tools
 * 
 * Provides MCP tools for checking EEG brainwave identity model status,
 * running enrollment, running authorization, and inspecting the EEG profile.
 * 
 * These tools work with the Python scripts in scripts/eeg_identity/
 * and the model artifacts in models/eeg_identity/.
 */

import fs from "fs";
import path from "path";
import { logger } from "../utils/logger";
import { getUserDataPath } from "../utils/userContext";
import { handlePipelineRun } from "./pipelineTools";

const PROJECT_ROOT = process.cwd();

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface EegConfig {
  mode: string;
  device_type: string;
  sample_rate: number;
  num_channels: number;
  channels: string[];
  num_tasks: number;
  num_valid_tasks: number;
  feature_dim: number;
  model_type: string;
  frequency_bands: Record<string, number[]>;
  created_at: string;
  statistics: {
    num_samples: number;
    feature_dim: number;
    mean_similarity: number;
    std_similarity: number;
    similarity_threshold_1std: number;
    similarity_threshold_2std: number;
    percentiles: Record<string, number>;
  };
}

interface EegModelStatusResult {
  available: boolean;
  model_type?: string;
  created_at?: string;
  num_channels?: number;
  sample_rate?: number;
  feature_dim?: number;
  num_enrollment_samples?: number;
  mean_similarity?: number;
  similarity_threshold?: number;
  message: string;
}

interface EegProfileSummaryResult {
  available: boolean;
  config?: Partial<EegConfig>;
  spectral_summary?: Record<string, Record<string, { mean_relative_power: number; std_relative_power: number }>>;
  enrollment_tasks?: Array<{ task: string; quality_passed: boolean; category?: string }>;
  statistics?: Record<string, any>;
  message: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function getEegModelsDir(userId: string | null): string {
  return getUserDataPath(path.join(PROJECT_ROOT, "models", "eeg_identity"), userId);
}

function loadEegConfig(userId: string | null): EegConfig | null {
  const modelsDir = getEegModelsDir(userId);
  const configPath = path.join(modelsDir, "config.json");
  
  if (!fs.existsSync(configPath)) {
    return null;
  }
  
  try {
    const raw = fs.readFileSync(configPath, "utf-8");
    return JSON.parse(raw) as EegConfig;
  } catch (e) {
    logger.error("Failed to load EEG identity config", { error: String(e) });
    return null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool handlers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Check if the EEG identity model is trained and available.
 */
export async function handleEegModelStatus(
  userId: string | null = null
): Promise<EegModelStatusResult> {
  const config = loadEegConfig(userId);
  
  if (!config) {
    return {
      available: false,
      message: "No EEG identity model found. Run enroll_brainwaves.py to create one.",
    };
  }
  
  const stats = config.statistics || {};
  
  return {
    available: true,
    model_type: config.model_type || "centroid",
    created_at: config.created_at,
    num_channels: config.num_channels,
    sample_rate: config.sample_rate,
    feature_dim: stats.feature_dim || config.feature_dim,
    num_enrollment_samples: stats.num_samples,
    mean_similarity: stats.mean_similarity,
    similarity_threshold: stats.similarity_threshold_1std,
    message: `EEG identity model available. Enrolled with ${stats.num_samples || 0} samples across ${config.num_valid_tasks || 0} tasks.`,
  };
}

/**
 * Run the enrollment script via the pipeline system.
 * Accepts mode (synthetic or hid) and optional serial number.
 */
export async function handleEegEnroll(
  { mode, serial, task_duration }: {
    mode?: string;
    serial?: string;
    task_duration?: number;
  },
  userId: string | null = null
): Promise<any> {
  const args: string[] = [];
  
  if (mode === "synthetic" || (!serial && !mode)) {
    args.push("--synthetic");
  } else if (serial) {
    args.push("--serial", serial);
  }
  
  if (task_duration) {
    args.push("--task-duration", String(task_duration));
  }
  
  logger.info("Starting EEG enrollment via pipeline", { args });
  return handlePipelineRun({ script: "enroll_brainwaves", args }, userId);
}

/**
 * Run the authorization script via the pipeline system.
 * Returns the assurance signal result.
 */
export async function handleEegAuthorize(
  { mode, serial, window_seconds }: {
    mode?: string;
    serial?: string;
    window_seconds?: number;
  },
  userId: string | null = null
): Promise<any> {
  const args: string[] = ["--json"];  // Always get JSON output for MCP
  
  if (mode === "synthetic" || (!serial && !mode)) {
    args.push("--synthetic");
  } else if (serial) {
    args.push("--serial", serial);
  }
  
  if (window_seconds) {
    args.push("--window", String(window_seconds));
  }
  
  logger.info("Starting EEG authorization via pipeline", { args });
  return handlePipelineRun({ script: "authorize_brainwaves", args }, userId);
}

/**
 * Get a summary of the trained EEG identity profile.
 * Shows spectral profile, enrollment tasks, and statistics.
 */
export async function handleEegProfileSummary(
  userId: string | null = null
): Promise<EegProfileSummaryResult> {
  const modelsDir = getEegModelsDir(userId);
  const config = loadEegConfig(userId);
  
  if (!config) {
    return {
      available: false,
      message: "No EEG identity model found. Run enroll_brainwaves.py to create one.",
    };
  }
  
  // Load spectral profile
  let spectralSummary: Record<string, Record<string, { mean_relative_power: number; std_relative_power: number }>> | undefined;
  const spectralPath = path.join(modelsDir, "spectral_profile.json");
  if (fs.existsSync(spectralPath)) {
    try {
      const raw = JSON.parse(fs.readFileSync(spectralPath, "utf-8"));
      // Extract the aggregate summary with both mean and std
      const aggregate = raw.aggregate || {};
      spectralSummary = {};
      for (const [channel, bands] of Object.entries(aggregate)) {
        spectralSummary[channel] = {};
        for (const [band, values] of Object.entries(bands as Record<string, any>)) {
          spectralSummary[channel][band] = {
            mean_relative_power: values.mean_relative_power ?? 0,
            std_relative_power: values.std_relative_power ?? 0,
          };
        }
      }
    } catch (e) {
      logger.warn("Failed to load spectral profile", { error: String(e) });
    }
  }
  
  // Load enrollment log
  let enrollmentTasks: Array<{ task: string; quality_passed: boolean; category?: string }> | undefined;
  const logPath = path.join(modelsDir, "enrollment_log.json");
  if (fs.existsSync(logPath)) {
    try {
      const log = JSON.parse(fs.readFileSync(logPath, "utf-8"));
      enrollmentTasks = log.map((entry: any) => ({
        task: entry.task || entry.label,
        quality_passed: entry.quality_passed,
        category: entry.category,
      }));
    } catch (e) {
      logger.warn("Failed to load enrollment log", { error: String(e) });
    }
  }
  
  return {
    available: true,
    config: {
      mode: config.mode,
      device_type: config.device_type,
      sample_rate: config.sample_rate,
      num_channels: config.num_channels,
      channels: config.channels,
      num_tasks: config.num_tasks,
      num_valid_tasks: config.num_valid_tasks,
      feature_dim: config.feature_dim,
      model_type: config.model_type,
      created_at: config.created_at,
    },
    spectral_summary: spectralSummary,
    enrollment_tasks: enrollmentTasks,
    statistics: config.statistics,
    message: `EEG profile: ${config.num_channels} channels, ${config.num_valid_tasks} tasks, ${config.statistics?.num_samples || 0} feature samples.`,
  };
}
