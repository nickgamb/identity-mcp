import { execSync } from "child_process";
import { logger } from "./logger";

/**
 * Find available Python executable (Windows: python first; Unix: python3 first).
 */
export function getPythonCommand(): string {
  const isWindows = process.platform === "win32";
  const commands = isWindows
    ? ["python", "python3", "py"]
    : ["python3", "python", "py"];

  for (const cmd of commands) {
    try {
      execSync(`${cmd} --version`, { stdio: "ignore" });
      return cmd;
    } catch {
      // try next
    }
  }

  return "python3";
}

export const PYTHON_CMD = getPythonCommand();
logger.info(`Using Python command: ${PYTHON_CMD}`);
