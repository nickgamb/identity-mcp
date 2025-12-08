/**
 * Pipeline Tools - Run processing scripts via MCP
 * 
 * Whitelisted scripts only - no arbitrary code execution.
 * Can be triggered by dashboard UI or by AI via MCP tools.
 */

import { spawn } from "child_process";
import path from "path";
import { config } from "../config";
import { logger } from "../utils/logger";

// Whitelisted scripts that can be run
const ALLOWED_SCRIPTS: Record<string, { path: string; description: string }> = {
  parse_conversations: {
    path: "scripts/conversation_processing/parse_conversations.py",
    description: "Parse raw conversations.json into structured JSONL files",
  },
  analyze_patterns: {
    path: "scripts/conversation_processing/analyze_patterns.py",
    description: "Discover distinctive terms, topics, and patterns",
  },
  parse_memories: {
    path: "scripts/conversation_processing/parse_memories.py",
    description: "Convert ChatGPT memories.json to searchable context",
  },
  analyze_identity: {
    path: "scripts/conversation_processing/analyze_identity.py",
    description: "Extract relational patterns and identity markers",
  },
  build_emergence_map: {
    path: "scripts/conversation_processing/build_emergence_map.py",
    description: "Index conversations and identify key events",
  },
  train_identity_model: {
    path: "scripts/identity_model/train_identity_model.py",
    description: "Train the semantic identity embedding model",
  },
};

interface ScriptResult {
  success: boolean;
  script: string;
  output: string[];
  exitCode: number | null;
  duration: number;
  error?: string;
}

/**
 * List available pipeline scripts
 */
export async function handlePipelineList(): Promise<{
  scripts: Array<{ id: string; path: string; description: string }>;
}> {
  return {
    scripts: Object.entries(ALLOWED_SCRIPTS).map(([id, info]) => ({
      id,
      path: info.path,
      description: info.description,
    })),
  };
}

/**
 * Run a pipeline script
 */
export async function handlePipelineRun({
  script,
  args = [],
}: {
  script: string;
  args?: string[];
}): Promise<ScriptResult> {
  const startTime = Date.now();

  // Validate script is allowed
  const scriptInfo = ALLOWED_SCRIPTS[script];
  if (!scriptInfo) {
    return {
      success: false,
      script,
      output: [`Error: Unknown script '${script}'`],
      exitCode: null,
      duration: 0,
      error: `Script '${script}' is not in the allowed list. Use pipeline_list to see available scripts.`,
    };
  }

  const scriptPath = path.join(config.PROJECT_ROOT, scriptInfo.path);
  logger.info("Running pipeline script", { script, scriptPath, args });

  return new Promise((resolve) => {
    const output: string[] = [];
    output.push(`$ python ${scriptInfo.path} ${args.join(" ")}`);
    output.push("");

    const proc = spawn("python", [scriptPath, ...args], {
      cwd: config.PROJECT_ROOT,
      env: { ...process.env },
    });

    proc.stdout.on("data", (data) => {
      const lines = data.toString().split("\n").filter((l: string) => l);
      output.push(...lines);
    });

    proc.stderr.on("data", (data) => {
      const lines = data.toString().split("\n").filter((l: string) => l);
      output.push(...lines.map((l: string) => `[stderr] ${l}`));
    });

    proc.on("error", (error) => {
      const duration = Date.now() - startTime;
      output.push(`[error] Failed to start: ${error.message}`);
      resolve({
        success: false,
        script,
        output,
        exitCode: null,
        duration,
        error: error.message,
      });
    });

    proc.on("close", (code) => {
      const duration = Date.now() - startTime;
      output.push("");
      output.push(`Process exited with code ${code} (${(duration / 1000).toFixed(1)}s)`);

      logger.info("Pipeline script completed", { script, exitCode: code, duration });

      resolve({
        success: code === 0,
        script,
        output,
        exitCode: code,
        duration,
      });
    });
  });
}

/**
 * Get status of a running script (placeholder for future)
 */
export async function handlePipelineStatus({
  script,
}: {
  script: string;
}): Promise<{ running: boolean; script: string }> {
  // TODO: Track running scripts and their status
  return {
    running: false,
    script,
  };
}

/**
 * Run all pipeline scripts in order
 */
export async function handlePipelineRunAll(): Promise<{
  success: boolean;
  results: ScriptResult[];
  totalDuration: number;
}> {
  const startTime = Date.now();
  const results: ScriptResult[] = [];
  const order = [
    "parse_conversations",
    "analyze_patterns",
    "parse_memories",
    "analyze_identity",
    "build_emergence_map",
  ];

  for (const script of order) {
    const result = await handlePipelineRun({ script });
    results.push(result);

    // Stop on first failure
    if (!result.success) {
      break;
    }
  }

  return {
    success: results.every((r) => r.success),
    results,
    totalDuration: Date.now() - startTime,
  };
}

