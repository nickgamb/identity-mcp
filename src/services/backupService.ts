import fs from "fs";
import path from "path";
import { execSync } from "child_process";
import { config } from "../config";
import { logger } from "../utils/logger";

// ── Types ──────────────────────────────────────────────────────────────

export interface BackupConfig {
  enabled: boolean;
  intervalHours: number;
  retentionDays: number; // 0 = keep all (no pruning)
  includeCorpus: boolean; // also tar conversations + files + memory
}

export interface BackupStatus {
  config: BackupConfig;
  lastBackupTime: string | null;
  backupCount: number;
  nextBackupIn: string | null;
  backupDir: string;
  includesPostgres: boolean;
}

// ── Module state ───────────────────────────────────────────────────────

const CONFIG_PATH = path.join(config.MEMORY_DIR, "backup-config.json");
const BACKUP_DIR = path.join(config.MEMORY_DIR, "backups");

let backupConfig: BackupConfig = {
  enabled: false,
  intervalHours: 24,
  retentionDays: 0,
  includeCorpus: false,
};

let lastBackupTime: number | null = null;
let loopTimer: ReturnType<typeof setInterval> | null = null;

// ── Config persistence ─────────────────────────────────────────────────

function loadConfig(): void {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const raw = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8"));
      if (typeof raw.enabled === "boolean") backupConfig.enabled = raw.enabled;
      if (typeof raw.intervalHours === "number" && raw.intervalHours >= 1) {
        backupConfig.intervalHours = raw.intervalHours;
      }
      if (typeof raw.retentionDays === "number" && raw.retentionDays >= 0) {
        backupConfig.retentionDays = raw.retentionDays;
      }
      if (typeof raw.includeCorpus === "boolean") {
        backupConfig.includeCorpus = raw.includeCorpus;
      }
      if (typeof raw.lastBackupTime === "number") {
        lastBackupTime = raw.lastBackupTime;
      }
      logger.info("Backup config loaded", backupConfig);
    }
  } catch (e) {
    logger.warn("Failed to load backup config, using defaults", {
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
          enabled: backupConfig.enabled,
          intervalHours: backupConfig.intervalHours,
          retentionDays: backupConfig.retentionDays,
          includeCorpus: backupConfig.includeCorpus,
          lastBackupTime,
        },
        null,
        2
      ),
      "utf-8"
    );
  } catch (e) {
    logger.warn("Failed to save backup config", { error: String(e) });
  }
}

// ── Backup logic ──────────────────────────────────────────────────────

function shouldBackup(): boolean {
  if (!backupConfig.enabled) return false;
  if (lastBackupTime !== null) {
    const elapsed = Date.now() - lastBackupTime;
    const intervalMs = backupConfig.intervalHours * 60 * 60 * 1000;
    if (elapsed < intervalMs) return false;
  }
  return true;
}

function dumpLettaPostgres(backupPath: string): boolean {
  const pgHost = process.env.LETTA_PG_HOST || "letta-postgres";
  const pgUser = process.env.LETTA_PG_USER || "letta";
  const pgPass = process.env.LETTA_PG_PASS || "letta";
  const pgDb = process.env.LETTA_PG_DB || "letta";
  const dumpFile = path.join(backupPath, "letta-db.sql");

  try {
    // Write directly to file (-f) — avoids buffering huge dumps in Node (ENOBUFS).
    execSync(
      `pg_dump -h ${pgHost} -U ${pgUser} -d ${pgDb} --no-password -f "${dumpFile}"`,
      {
        env: { ...process.env, PGPASSWORD: pgPass },
        timeout: 600_000,
        stdio: ["pipe", "pipe", "pipe"],
      }
    );
    const stat = fs.statSync(dumpFile);
    if (stat.size === 0) {
      throw new Error("pg_dump produced an empty file");
    }
    logger.info("Letta postgres dumped", {
      file: dumpFile,
      sizeBytes: stat.size,
    });
    return true;
  } catch (e) {
    logger.error("Failed to dump Letta postgres", { error: String(e) });
    try {
      if (fs.existsSync(dumpFile)) fs.unlinkSync(dumpFile);
    } catch {}
    try {
      fs.writeFileSync(
        path.join(backupPath, "letta-db.FAILED"),
        `pg_dump failed at ${new Date().toISOString()}\n${String(e)}`,
        "utf-8"
      );
    } catch {}
    return false;
  }
}

function archiveCorpus(backupPath: string): boolean {
  const archiveFile = path.join(backupPath, "corpus.tar.gz");
  const appRoot = config.PROJECT_ROOT;
  const candidates: Array<{ rel: string; abs: string }> = [
    { rel: "conversations", abs: path.join(appRoot, "conversations") },
    { rel: "files", abs: config.FILES_DIR },
    { rel: "memory", abs: config.MEMORY_DIR },
  ];
  const existingDirs = candidates
    .filter((c) => fs.existsSync(c.abs))
    .map((c) => c.rel);

  if (existingDirs.length === 0) {
    logger.warn("No corpus directories found to archive", {
      appRoot,
      checked: candidates.map((c) => c.abs),
    });
    return false;
  }

  const excludePattern = "--exclude=memory/backups";

  try {
    execSync(
      `tar -czf "${archiveFile}" ${excludePattern} ${existingDirs.join(" ")}`,
      {
        cwd: appRoot,
        timeout: 300_000, // 5 min for large corpus
        stdio: ["pipe", "pipe", "pipe"],
      }
    );
    const stat = fs.statSync(archiveFile);
    logger.info("Corpus archived", {
      file: archiveFile,
      sizeMB: Math.round(stat.size / 1024 / 1024 * 10) / 10,
      dirs: existingDirs,
    });
    return true;
  } catch (e) {
    logger.error("Failed to archive corpus", { error: String(e) });
    try {
      fs.writeFileSync(
        path.join(backupPath, "corpus.FAILED"),
        `tar failed at ${new Date().toISOString()}\n${String(e)}`,
        "utf-8"
      );
    } catch {}
    return false;
  }
}

function createBackup(): boolean {
  try {
    if (!fs.existsSync(BACKUP_DIR)) {
      fs.mkdirSync(BACKUP_DIR, { recursive: true });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const backupPath = path.join(BACKUP_DIR, timestamp);
    fs.mkdirSync(backupPath, { recursive: true });

    // Always dump Letta postgres (the critical data)
    const pgSuccess = dumpLettaPostgres(backupPath);

    // Optionally archive full corpus
    let corpusSuccess = true;
    if (backupConfig.includeCorpus) {
      corpusSuccess = archiveCorpus(backupPath);
    }

    if (pgSuccess) {
      lastBackupTime = Date.now();
      saveConfig();
      logger.info("Backup created", {
        path: backupPath,
        pgDumped: true,
        corpusArchived: backupConfig.includeCorpus ? corpusSuccess : "skipped",
      });
    } else {
      logger.error("Backup failed (pg_dump)", { path: backupPath });
    }
    if (backupConfig.includeCorpus && !corpusSuccess) {
      logger.error("Corpus archive failed — see corpus.FAILED in backup folder");
    }
    return pgSuccess;
  } catch (e) {
    logger.error("Failed to create backup", { error: String(e) });
    return false;
  }
}

function pruneBackups(): void {
  if (backupConfig.retentionDays <= 0) return;
  if (!fs.existsSync(BACKUP_DIR)) return;

  const cutoff = Date.now() - backupConfig.retentionDays * 24 * 60 * 60 * 1000;
  const entries = fs.readdirSync(BACKUP_DIR);
  let pruned = 0;

  for (const entry of entries) {
    const entryPath = path.join(BACKUP_DIR, entry);
    const stat = fs.statSync(entryPath);
    if (!stat.isDirectory()) continue;

    // Parse timestamp from directory name (ISO format with dashes)
    const dateStr = entry.replace(/-(\d{2})-(\d{2})-(\d+)Z$/, ":$1:$2.$3Z")
      .replace(/T(\d{2})-/, "T$1:");
    const entryTime = new Date(dateStr).getTime();

    if (!isNaN(entryTime) && entryTime < cutoff) {
      fs.rmSync(entryPath, { recursive: true, force: true });
      pruned++;
    }
  }

  if (pruned > 0) {
    logger.info("Pruned old backups", { pruned, retentionDays: backupConfig.retentionDays });
  }
}

function getBackupCount(): number {
  if (!fs.existsSync(BACKUP_DIR)) return 0;
  return fs.readdirSync(BACKUP_DIR).filter(e => {
    return fs.statSync(path.join(BACKUP_DIR, e)).isDirectory();
  }).length;
}

function checkLoop(): void {
  if (!shouldBackup()) return;
  createBackup();
  pruneBackups();
}

// ── Exported API ───────────────────────────────────────────────────────

export function getBackupStatus(): BackupStatus {
  const nextBackupIn = (() => {
    if (!backupConfig.enabled) return null;
    if (lastBackupTime === null) return "Now (pending first backup)";
    const intervalMs = backupConfig.intervalHours * 60 * 60 * 1000;
    const remaining = intervalMs - (Date.now() - lastBackupTime);
    if (remaining <= 0) return "Now";
    const hours = Math.floor(remaining / 3600000);
    const mins = Math.floor((remaining % 3600000) / 60000);
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  })();

  return {
    config: { ...backupConfig },
    lastBackupTime: lastBackupTime ? new Date(lastBackupTime).toISOString() : null,
    backupCount: getBackupCount(),
    nextBackupIn,
    backupDir: BACKUP_DIR,
    includesPostgres: true,
  };
}

export function updateBackupConfig(
  patch: Partial<BackupConfig>
): { success: boolean; config: BackupConfig } {
  if (typeof patch.enabled === "boolean") {
    backupConfig.enabled = patch.enabled;
  }
  if (typeof patch.intervalHours === "number") {
    backupConfig.intervalHours = Math.max(1, Math.min(168, patch.intervalHours));
  }
  if (typeof patch.retentionDays === "number") {
    backupConfig.retentionDays = Math.max(0, Math.min(365, patch.retentionDays));
  }
  if (typeof patch.includeCorpus === "boolean") {
    backupConfig.includeCorpus = patch.includeCorpus;
  }
  saveConfig();
  logger.info("Backup config updated", backupConfig);
  return { success: true, config: { ...backupConfig } };
}

export function triggerBackupNow(): {
  success: boolean;
  message: string;
  corpusArchived?: boolean;
} {
  const ok = createBackup();
  if (ok) pruneBackups();
  const corpusNote = backupConfig.includeCorpus
    ? " + corpus.tar.gz (if no corpus.FAILED marker)"
    : "";
  return ok
    ? {
        success: true,
        message: `Backup saved under memory/backups/ (letta-db.sql${corpusNote})`,
        corpusArchived: backupConfig.includeCorpus,
      }
    : {
        success: false,
        message:
          "Letta postgres backup failed (check letta-postgres is up, mcp-server has pg_dump, and logs for letta-db.FAILED)",
      };
}

export function startBackupLoop(): void {
  loadConfig();
  if (loopTimer) return;
  // Check every 5 minutes
  loopTimer = setInterval(() => {
    try {
      checkLoop();
    } catch (e) {
      logger.error("Backup check loop error", { error: String(e) });
    }
  }, 5 * 60_000);
  logger.info("Backup loop started", {
    enabled: backupConfig.enabled,
    intervalHours: backupConfig.intervalHours,
  });
}

export function stopBackupLoop(): void {
  if (loopTimer) {
    clearInterval(loopTimer);
    loopTimer = null;
    logger.info("Backup loop stopped");
  }
}
