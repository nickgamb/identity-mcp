import fs from "fs";
import path from "path";
import { logger } from "../utils/logger";
import { config } from "../config";
import { getUserDataPath, ensureUserDirectory } from "../utils/userContext";
import {
  isTabularFile,
  loadTabularCorpusText,
  searchTabularRows,
} from "../utils/csvCorpus";

export interface FileDocument {
  filename: string;
  filepath: string; // Full path relative to files directory
  title?: string;
  content: string;
  fileNumber?: number; // For numbered files like 001_foundation.txt
  category?: string; // Determined by folder or content
  folder?: string; // Which subfolder it's in
  extension?: string; // File extension
  metadata?: {
    title?: string;
    purpose?: string;
    [key: string]: string | undefined;
  };
}

/**
 * Generic file loader for the files/ directory (RAG storage)
 * Loads all files recursively - no hardcoded folder names
 * Files are raw content used for training, RAG, and context
 */
export class FileLoader {
  private filesDir: string;
  private userId: string | null;

  constructor(filesDir?: string, userId: string | null = null) {
    const baseDir = filesDir || config.FILES_DIR;
    this.filesDir = getUserDataPath(baseDir, userId);
    this.userId = userId;
    ensureUserDirectory(this.filesDir);
  }

  /**
   * Lists all files in directory, optionally filtered by folder
   */
  async listFiles(folder?: string): Promise<string[]> {
    try {
      // Normalize folder - prevent duplication if folder matches filesDir base name
      let targetDir = this.filesDir;
      if (folder) {
        const normalizedFolder = folder.replace(/^\/+|\/+$/g, '');
        const filesDirBase = path.basename(this.filesDir);
        // If folder is the same as filesDir base name, don't join (already in filesDir)
        if (normalizedFolder !== '' && normalizedFolder !== filesDirBase) {
          targetDir = path.join(this.filesDir, normalizedFolder);
        }
      }

      if (!fs.existsSync(targetDir)) {
        logger.warn("Files directory not found", { dir: targetDir });
        return [];
      }

      const files: string[] = [];

      if (folder) {
        const normalizedFolder = folder.replace(/^\/+|\/+$/g, '');
        const filesDirBase = path.basename(this.filesDir);
        // If folder matches filesDir base name, list all files recursively
        if (normalizedFolder === '' || normalizedFolder === filesDirBase) {
          await this.listFilesRecursive(this.filesDir, "", files);
        } else {
          const folderFiles = await fs.promises.readdir(targetDir);
          files.push(...folderFiles
            .filter(f => this.isValidFile(f))
            .map(f => path.join(normalizedFolder, f))
          );
        }
      } else {
        await this.listFilesRecursive(this.filesDir, "", files);
      }

      // Sort by file number if present, otherwise alphabetically
      files.sort((a, b) => {
        const numA = this.extractFileNumber(a);
        const numB = this.extractFileNumber(b);
        if (numA !== null && numB !== null) return numA - numB;
        if (numA !== null) return -1;
        if (numB !== null) return 1;
        return a.localeCompare(b);
      });

      return files;
    } catch (error) {
      logger.error("Error listing files", error);
      return [];
    }
  }

  /** Extensions we treat as binary / non-text for listing and RAG load */
  private static readonly BLOCKED_EXTENSIONS = new Set([
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".gz", ".tar", ".7z", ".rar", ".bz2",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".wasm",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
  ]);

  /**
   * Include user-uploaded documents: all non-hidden files except known binary types.
   * (Previously only .txt/.md/.json/etc., which hid .csv and other uploads.)
   */
  private isValidFile(filename: string): boolean {
    if (filename.includes("Zone.Identifier")) return false;
    if (filename.startsWith(".")) return false;
    const ext = path.extname(filename).toLowerCase();
    if (FileLoader.BLOCKED_EXTENSIONS.has(ext)) return false;
    return true;
  }

  /**
   * Recursively lists all files in directory tree
   */
  private async listFilesRecursive(dir: string, relativePath: string, files: string[]): Promise<void> {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      if (entry.name.startsWith(".")) continue;
      if (entry.name.includes("Zone.Identifier")) continue;
      
      const fullPath = path.join(dir, entry.name);
      const relPath = relativePath ? path.join(relativePath, entry.name) : entry.name;

      if (entry.isDirectory()) {
        await this.listFilesRecursive(fullPath, relPath, files);
      } else if (entry.isFile() && this.isValidFile(entry.name)) {
        files.push(relPath);
      }
    }
  }

  /**
   * Metadata for a file path without reading content (for listing).
   */
  describeFile(filepath: string): Omit<FileDocument, "content"> {
    const fileNumber = this.extractFileNumber(filepath);
    const folder = path.dirname(filepath);
    const filename = path.basename(filepath);
    const extension = path.extname(filepath);
    const category = this.categorizeFile(filepath);

    return {
      filename,
      filepath,
      fileNumber: fileNumber ?? undefined,
      category,
      folder: folder !== "." ? folder : undefined,
      extension,
    };
  }

  /**
   * Read file text; CSV/TSV use csv_corpus row formatting (same as Letta ingest).
   */
  private async readFileContent(fullPath: string, filepath: string): Promise<string> {
    if (isTabularFile(filepath)) {
      const corpus = await loadTabularCorpusText(fullPath);
      if (corpus !== null) {
        return corpus;
      }
    }
    return fs.promises.readFile(fullPath, "utf8");
  }

  /**
   * Loads a single file by its path (relative to files directory)
   */
  async loadFile(filepath: string): Promise<FileDocument | null> {
    try {
      const fullPath = path.join(this.filesDir, filepath);
      
      if (!fs.existsSync(fullPath)) {
        logger.warn("File not found", { file: fullPath });
        return null;
      }

      const content = await this.readFileContent(fullPath, filepath);
      const fileNumber = this.extractFileNumber(filepath);
      const folder = path.dirname(filepath);
      const filename = path.basename(filepath);
      const extension = path.extname(filepath);
      const category = this.categorizeFile(filepath);
      const metadata = this.extractMetadata(content);

      return {
        filename,
        filepath,
        title: metadata?.title,
        content,
        fileNumber: fileNumber ?? undefined,
        category,
        folder: folder !== "." ? folder : undefined,
        extension,
        metadata,
      };
    } catch (error) {
      logger.error("Error loading file", { filepath, error });
      return null;
    }
  }

  /**
   * Loads all files from a specific folder
   */
  async loadFilesFromFolder(folder: string): Promise<FileDocument[]> {
    // Normalize folder - remove leading/trailing slashes and prevent "files" duplication
    const normalizedFolder = folder.replace(/^\/+|\/+$/g, '');
    if (normalizedFolder === '' || normalizedFolder === path.basename(this.filesDir)) {
      // If folder is empty or matches the base files directory name, list all files
      return this.loadAllFiles();
    }
    
    const filenames = await this.listFiles(normalizedFolder);
    const files: FileDocument[] = [];

    for (const filepath of filenames) {
      const file = await this.loadFile(filepath);
      if (file) {
        files.push(file);
      }
    }

    return files;
  }

  /**
   * Loads all files (recursively)
   */
  async loadAllFiles(): Promise<FileDocument[]> {
    const filenames = await this.listFiles();
    const files: FileDocument[] = [];

    for (const filepath of filenames) {
      const file = await this.loadFile(filepath);
      if (file) {
        files.push(file);
      }
    }

    return files;
  }

  /**
   * Gets numbered files (001-N) from any folder or all folders
   * @param folder Optional folder to limit search
   * @param maxNumber Maximum file number to include (default 10)
   */
  async getNumberedFiles(folder?: string, maxNumber: number = 10): Promise<FileDocument[]> {
    const all = folder 
      ? await this.loadFilesFromFolder(folder)
      : await this.loadAllFiles();
    
    return all.filter(f => 
      f.fileNumber !== undefined && 
      f.fileNumber >= 1 && 
      f.fileNumber <= maxNumber
    ).sort((a, b) => (a.fileNumber ?? 0) - (b.fileNumber ?? 0));
  }

  /**
   * Gets files by category (derived from folder path)
   */
  async getFilesByCategory(category: string): Promise<FileDocument[]> {
    const filenames = await this.listFiles();
    const files: FileDocument[] = [];
    for (const filepath of filenames) {
      if (this.categorizeFile(filepath) !== category) continue;
      const file = await this.loadFile(filepath);
      if (file) files.push(file);
    }
    return files;
  }

  /**
   * Lists all folders in the files directory
   */
  async listFolders(): Promise<string[]> {
    try {
      if (!fs.existsSync(this.filesDir)) {
        return [];
      }
      
      const entries = await fs.promises.readdir(this.filesDir, { withFileTypes: true });
      return entries
        .filter(e => e.isDirectory() && !e.name.startsWith("."))
        .map(e => e.name);
    } catch (error) {
      logger.error("Error listing folders", error);
      return [];
    }
  }

  /**
   * Searches files by content. Tabular files return matching rows only, not the full file.
   */
  async searchFiles(
    query: string,
    folder?: string,
    maxFiles: number = 20,
    maxRowsPerTabular: number = 10
  ): Promise<FileDocument[]> {
    const normalizedFolder = folder?.replace(/^\/+|\/+$/g, "");
    const filenames = normalizedFolder
      ? await this.listFiles(normalizedFolder)
      : await this.listFiles();

    const lowerQuery = query.toLowerCase().trim();
    if (!lowerQuery) return [];

    const results: FileDocument[] = [];

    for (const filepath of filenames) {
      if (results.length >= maxFiles) break;

      const fullPath = path.join(this.filesDir, filepath);
      if (!fs.existsSync(fullPath)) continue;

      const meta = this.describeFile(filepath);
      const nameMatch =
        meta.filename.toLowerCase().includes(lowerQuery) ||
        meta.filepath.toLowerCase().includes(lowerQuery);

      if (isTabularFile(filepath)) {
        const rowBlocks = await searchTabularRows(fullPath, query, maxRowsPerTabular);
        if (rowBlocks && rowBlocks.length > 0) {
          results.push({
            ...meta,
            content: rowBlocks.join("\n\n"),
            title: meta.filename,
          });
          continue;
        }
        if (nameMatch) {
          const file = await this.loadFile(filepath);
          if (file) {
            const preview = file.content.slice(0, 8000);
            results.push({
              ...file,
              content:
                preview.length < file.content.length
                  ? `${preview}\n\n[… truncated; use file_get for full corpus]`
                  : file.content,
            });
          }
        }
        continue;
      }

      const content = await fs.promises.readFile(fullPath, "utf8");
      const metadata = this.extractMetadata(content);
      const titleMatch = metadata?.title?.toLowerCase().includes(lowerQuery);
      if (
        content.toLowerCase().includes(lowerQuery) ||
        titleMatch ||
        nameMatch
      ) {
        const snippet = this.extractSearchSnippet(content, lowerQuery);
        results.push({
          ...meta,
          title: metadata?.title,
          content: snippet,
          metadata,
        });
      }
    }

    return results;
  }

  /** Surround the first match with local context for non-tabular text search hits. */
  private extractSearchSnippet(content: string, lowerQuery: string, radius = 400): string {
    const idx = content.toLowerCase().indexOf(lowerQuery);
    if (idx < 0) {
      return content.length > 2000 ? `${content.slice(0, 2000)}\n\n[… truncated]` : content;
    }
    const start = Math.max(0, idx - radius);
    const end = Math.min(content.length, idx + lowerQuery.length + radius);
    let snippet = content.slice(start, end);
    if (start > 0) snippet = `…${snippet}`;
    if (end < content.length) snippet = `${snippet}…`;
    return snippet;
  }

  /**
   * Extracts file number from filepath
   */
  private extractFileNumber(filepath: string): number | null {
    // Match patterns like "001_", "_001_", "_001."
    const match = filepath.match(/(?:^|[/_])(\d{3})(?:_|\.)/);
    if (match) {
      return parseInt(match[1], 10);
    }
    return null;
  }

  /**
   * Categorizes file based on folder path (generic, not hardcoded)
   */
  private categorizeFile(filepath: string): string {
    const parts = filepath.split(/[/\\]/);
    if (parts.length > 1) {
      // Use the first folder as category
      return parts[0].toLowerCase();
    }
    return "root";
  }

  /**
   * Extracts metadata from file content
   */
  private extractMetadata(content: string): FileDocument["metadata"] {
    const metadata: FileDocument["metadata"] = {};
    const lines = content.split("\n").slice(0, 50);

    // Try to extract title
    for (const line of lines) {
      if (line.match(/^Title:|^title:|^#\s+/i)) {
        metadata.title = line.replace(/^(Title:|title:|#\s+)/i, "").trim();
        break;
      }
      // First non-empty, non-code line as title
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith("```") && !trimmed.startsWith("*") && 
          !trimmed.startsWith("{") && trimmed.length > 10 && trimmed.length < 100) {
        metadata.title = trimmed;
        break;
      }
    }

    // Try to extract purpose
    for (const line of lines) {
      if (line.match(/Purpose:|purpose:/i)) {
        metadata.purpose = line.replace(/^(Purpose:|purpose:)/i, "").trim();
        break;
      }
    }

    return metadata;
  }
}
