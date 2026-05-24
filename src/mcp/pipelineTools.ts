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
import { PYTHON_CMD } from "../utils/pythonCmd";

// Whitelisted scripts that can be run
const ALLOWED_SCRIPTS: Record<string, { path: string; description: string }> = {
  parse_conversations: {
    path: "scripts/conversation_processing/parse_all_conversations.py",
    description:
      "Parse ChatGPT and/or Claude uploads into structured JSONL (runs each parser if its upload exists)",
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
  build_interaction_map: {
    path: "scripts/conversation_processing/build_interaction_map.py",
    description: "Index conversations and identify human communication patterns",
  },
  train_identity_model: {
    path: "scripts/identity_model/train_identity_model.py",
    description: "Train the semantic identity embedding model",
  },
  enroll_brainwaves: {
    path: "scripts/eeg_identity/enroll_brainwaves.py",
    description: "Enroll brainwave identity via EMOTIV Epoc X EEG",
  },
  authorize_brainwaves: {
    path: "scripts/eeg_identity/authorize_brainwaves.py",
    description: "Test live EEG authorization against enrolled brainwave model",
  },
  letta_register_tools: {
    path: "letta/register_tools.py",
    description: "Attach identity-mcp HTTP tools to the Letta agent (idempotent)",
  },
  letta_ingest: {
    path: "letta/ingest.py",
    description: "Ingest conversations, memory, and files into Letta archival memory (incremental, deduped)",
  },
  letta_ingest_init: {
    path: "letta/ingest.py",
    description: "Agent self-init: explore archival memory and refresh persona/human blocks",
  },
  letta_bootstrap_persona: {
    path: "letta/bootstrap_agent.py",
    description: "Ensure agent exists and seed persona from memory/identity.jsonl (skips archival)",
  },
};

/** Default CLI args when script id implies a variant of the same file */
const SCRIPT_DEFAULT_ARGS: Record<string, string[]> = {
  letta_ingest_init: ["--init-only"],
  letta_bootstrap_persona: ["--skip-archival"],
};

function scriptEnv(
  scriptId: string,
  base: NodeJS.ProcessEnv,
  userId: string | null
): NodeJS.ProcessEnv {
  const env = { ...base };
  if (userId) {
    env.USER_ID = userId;
  }
  if (scriptId.startsWith("letta_")) {
    env.LETTA_BASE_URL = config.LETTA_BASE_URL;
    env.LETTA_AGENT_NAME = config.LETTA_AGENT_NAME;
    env.MCP_SERVER_URL =
      process.env.MCP_SERVER_URL || `http://127.0.0.1:${config.PORT}`;
    env.DATA_ROOT = process.env.DATA_ROOT || config.PROJECT_ROOT;
  }
  return env;
}

interface ScriptResult {
  success: boolean;
  script: string;
  output: string[];
  exitCode: number | null;
  duration: number;
  error?: string;
}

interface RunningScript {
  script: string;
  startTime: number;
  args: string[];
  process: any;
  output: string[];
  finished: boolean;
  exitCode: number | null;
}

// Track currently running scripts
const runningScripts = new Map<string, RunningScript>();

/**
 * List available pipeline scripts
 */
export async function handlePipelineList(userId: string | null = null): Promise<{
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
  args,
}: {
  script: string;
  args?: string[];
}, userId: string | null = null): Promise<ScriptResult> {
  const startTime = Date.now();
  const runArgs =
    args !== undefined && args.length > 0
      ? args
      : SCRIPT_DEFAULT_ARGS[script] || [];

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

  // If the script is already running, don't spawn a duplicate
  const existing = runningScripts.get(script);
  if (existing && !existing.finished) {
    logger.info("Pipeline script already running, skipping duplicate start", { script });
    // Return a pending promise that resolves when the existing run finishes
    return new Promise((resolve) => {
      const check = setInterval(() => {
        if (existing.finished) {
          clearInterval(check);
          resolve({
            success: existing.exitCode === 0,
            script,
            output: existing.output,
            exitCode: existing.exitCode,
            duration: Date.now() - existing.startTime,
          });
        }
      }, 500);
    });
  }

  const scriptPath = path.join(config.PROJECT_ROOT, scriptInfo.path);
  logger.info("Running pipeline script", { script, scriptPath, args: runArgs });

  return new Promise((resolve) => {
    const output: string[] = [];
    output.push(`$ ${PYTHON_CMD} ${scriptInfo.path} ${runArgs.join(" ")}`);
    output.push("");

    const env = scriptEnv(script, process.env, userId);

    const proc = spawn(PYTHON_CMD, [scriptPath, ...runArgs], {
      cwd: config.PROJECT_ROOT,
      env,
    });

    // Track running script (share the output array so SSE can read it)
    const runEntry: RunningScript = {
      script,
      startTime,
      args: runArgs,
      process: proc,
      output,
      finished: false,
      exitCode: null,
    };
    runningScripts.set(script, runEntry);

    proc.stdout.on("data", (data: Buffer) => {
      const lines = data.toString().split("\n").filter((l: string) => l);
      output.push(...lines);
    });

    proc.stderr.on("data", (data: Buffer) => {
      const lines = data.toString().split("\n").filter((l: string) => l);
      output.push(...lines.map((l: string) => `[stderr] ${l}`));
    });

    proc.on("error", (error: Error) => {
      const duration = Date.now() - startTime;
      output.push(`[error] Failed to start: ${error.message}`);
      
      // Mark finished so SSE clients see it, then clean up after a delay
      runEntry.finished = true;
      runEntry.exitCode = null;
      setTimeout(() => runningScripts.delete(script), 5000);
      
      resolve({
        success: false,
        script,
        output,
        exitCode: null,
        duration,
        error: error.message,
      });
    });

    proc.on("close", (code: number | null) => {
      const duration = Date.now() - startTime;
      output.push("");
      output.push(`Process exited with code ${code} (${(duration / 1000).toFixed(1)}s)`);

      logger.info("Pipeline script completed", { script, exitCode: code, duration });

      // Mark finished so SSE clients see it, then clean up after a delay
      runEntry.finished = true;
      runEntry.exitCode = code;
      setTimeout(() => runningScripts.delete(script), 5000);

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
 * Get status of a running script
 */
export async function handlePipelineStatus({
  script,
}: {
  script: string;
}, userId: string | null = null): Promise<{ 
  running: boolean; 
  script: string;
  startTime?: number;
  duration?: number;
  args?: string[];
}> {
  const runningScript = runningScripts.get(script);
  
  if (!runningScript) {
    return {
      running: false,
      script,
    };
  }

  return {
    running: true,
    script,
    startTime: runningScript.startTime,
    duration: Date.now() - runningScript.startTime,
    args: runningScript.args,
  };
}

/**
 * Get all running scripts
 */
export async function handlePipelineListRunning(userId: string | null = null): Promise<{
  running: Array<{
    script: string;
    startTime: number;
    duration: number;
    args: string[];
  }>;
}> {
  const running = Array.from(runningScripts.values()).map(s => ({
    script: s.script,
    startTime: s.startTime,
    duration: Date.now() - s.startTime,
    args: s.args,
  }));

  return { running };
}

/**
 * Stream pipeline script output via SSE.
 * Sends each new output line as an SSE `data:` frame.
 * Closes when the script finishes.
 */
export function handlePipelineStream(scriptId: string, res: any): void {
  // SSE headers
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const entry = runningScripts.get(scriptId);
  if (!entry) {
    res.write(`data: ${JSON.stringify({ event: "error", message: "Script not running" })}\n\n`);
    res.end();
    return;
  }

  let cursor = 0;
  const flush = () => {
    while (cursor < entry.output.length) {
      const line = entry.output[cursor];
      res.write(`data: ${JSON.stringify({ line, index: cursor })}\n\n`);
      cursor++;
    }
  };

  // Send any lines already buffered
  flush();

  // Poll for new lines (50ms for smooth updates)
  const interval = setInterval(() => {
    flush();

    if (entry.finished) {
      // Final flush + close event
      flush();
      res.write(`data: ${JSON.stringify({ event: "done", exitCode: entry.exitCode })}\n\n`);
      clearInterval(interval);
      res.end();
    }
  }, 50);

  // Clean up if client disconnects
  res.on("close", () => {
    clearInterval(interval);
  });
}

/**
 * Return buffered pipeline output as JSON (polling-friendly).
 * The frontend calls this repeatedly with an incrementing cursor.
 *
 * Returns `started: false` when the script hasn't registered yet,
 * so the frontend knows to keep waiting (vs. `done: true` = finished).
 */
export function handlePipelineOutput(
  scriptId: string,
  cursor: number
): { lines: Array<{ line: string; index: number }>; started: boolean; done: boolean; exitCode?: number } {
  const entry = runningScripts.get(scriptId);
  if (!entry) {
    // Script not yet registered — tell frontend to keep polling
    return { lines: [], started: false, done: false };
  }

  const newLines = entry.output.slice(cursor).map((line, i) => ({
    line,
    index: cursor + i,
  }));

  const result: { lines: Array<{ line: string; index: number }>; started: boolean; done: boolean; exitCode?: number } = {
    lines: newLines,
    started: true,
    done: entry.finished && cursor + newLines.length >= entry.output.length,
  };

  if (entry.finished) {
    result.exitCode = entry.exitCode;
  }

  return result;
}

/**
 * Stop a running pipeline script.
 */
export async function handlePipelineStop({
  script,
}: {
  script: string;
}): Promise<{ stopped: boolean; script: string; message: string }> {
  const entry = runningScripts.get(script);
  if (!entry) {
    return { stopped: false, script, message: "Script is not running" };
  }

  try {
    // On Windows, SIGTERM calls TerminateProcess which skips Python's finally blocks.
    // SIGINT simulates Ctrl+C, letting Python handle KeyboardInterrupt for clean shutdown.
    const signal = process.platform === "win32" ? "SIGINT" : "SIGTERM";
    entry.process.kill(signal);
    logger.info("Stopped pipeline script", { script, signal });
    return { stopped: true, script, message: "Script stopped" };
  } catch (e) {
    return { stopped: false, script, message: `Failed to stop: ${e}` };
  }
}

/**
 * Run all pipeline scripts in order
 */
export async function handlePipelineRunAll(userId: string | null = null): Promise<{
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
    "build_interaction_map",
  ];

  for (const script of order) {
    const result = await handlePipelineRun({ script }, userId);
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

