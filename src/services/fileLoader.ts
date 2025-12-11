import fs from "fs";
import path from "path";
import { logger } from "../utils/logger";
import { config } from "../config";
import { getUserDataPath, ensureUserDirectory } from "../utils/userContext";

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

  /**
   * Check if file should be included (skip system files)
   */
  private isValidFile(filename: string): boolean {
    if (filename.includes("Zone.Identifier")) return false;
    if (filename.startsWith(".")) return false;
    // Include common text/data formats
    const validExtensions = [".txt", ".md", ".json", ".jsonl", ".yaml", ".yml"];
    return validExtensions.some(ext => filename.endsWith(ext));
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
   * Loads a single file by its path (relative to files directory)
   */
  async loadFile(filepath: string): Promise<FileDocument | null> {
    try {
      const fullPath = path.join(this.filesDir, filepath);
      
      if (!fs.existsSync(fullPath)) {
        logger.warn("File not found", { file: fullPath });
        return null;
      }

      const content = await fs.promises.readFile(fullPath, "utf8");
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
    const all = await this.loadAllFiles();
    return all.filter(f => f.category === category);
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
   * Searches files by content
   */
  async searchFiles(query: string, folder?: string): Promise<FileDocument[]> {
    const files = folder 
      ? await this.loadFilesFromFolder(folder)
      : await this.loadAllFiles();
    
    const lowerQuery = query.toLowerCase();
    
    return files.filter(f => 
      f.content.toLowerCase().includes(lowerQuery) ||
      f.title?.toLowerCase().includes(lowerQuery) ||
      f.filename.toLowerCase().includes(lowerQuery) ||
      f.filepath.toLowerCase().includes(lowerQuery)
    );
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
