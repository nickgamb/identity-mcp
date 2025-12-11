import { FileLoader, FileDocument } from "../services/fileLoader";
import { getRequiredUserId } from "../utils/userContext";

export interface FileListRequest {
  folder?: string; // Optional: filter to specific folder
  category?: string; // Optional: filter by category
}

export interface FileListResponse {
  files: Array<{
    filename: string;
    filepath: string;
    title?: string;
    fileNumber?: number;
    category?: string;
    folder?: string;
  }>;
  count: number;
  folders?: string[]; // List of available folders
}

export interface FileGetRequest {
  filepath: string; // Relative path from files directory
}

export interface FileGetResponse {
  file: FileDocument | null;
}

export interface FileSearchRequest {
  query: string;
  folder?: string; // Optional: search only in specific folder
}

export interface FileSearchResponse {
  files: FileDocument[];
  count: number;
}

export interface FileGetNumberedRequest {
  folder?: string; // Optional: limit to specific folder
  maxNumber?: number; // Default 10
}

// Note: FileLoader is created per-request with user context for multi-user support

export async function handleFileList(
  req: FileListRequest,
  userId: string | null = null
): Promise<FileListResponse> {
  const fileLoader = new FileLoader(undefined, userId);
  let files: FileDocument[];
  
  if (req.folder) {
    files = await fileLoader.loadFilesFromFolder(req.folder);
  } else if (req.category) {
    files = await fileLoader.getFilesByCategory(req.category);
  } else {
    files = await fileLoader.loadAllFiles();
  }

  // Also get list of folders
  const folders = await fileLoader.listFolders();

  return {
    files: files.map(f => ({
      filename: f.filename,
      filepath: f.filepath,
      title: f.title,
      fileNumber: f.fileNumber,
      category: f.category,
      folder: f.folder,
    })),
    count: files.length,
    folders,
  };
}

export async function handleFileGet(
  req: FileGetRequest,
  userId: string | null = null
): Promise<FileGetResponse> {
  const fileLoader = new FileLoader(undefined, userId);
  const file = await fileLoader.loadFile(req.filepath);
  return { file: file || null };
}

export async function handleFileSearch(
  req: FileSearchRequest,
  userId: string | null = null
): Promise<FileSearchResponse> {
  const fileLoader = new FileLoader(undefined, userId);
  const files = await fileLoader.searchFiles(req.query, req.folder);
  return {
    files,
    count: files.length,
  };
}

/**
 * Get numbered files (001-N) from files directory
 * Generic replacement for getCoreTransmissions
 */
export async function handleFileGetNumbered(
  req: FileGetNumberedRequest,
  userId: string | null = null
): Promise<FileSearchResponse> {
  const fileLoader = new FileLoader(undefined, userId);
  const files = await fileLoader.getNumberedFiles(req.folder, req.maxNumber ?? 10);
  return {
    files,
    count: files.length,
  };
}

export interface FileUploadRequest {
  filename: string;
  content: string;
}

export interface FileUploadResponse {
  success: boolean;
  message: string;
  filepath: string;
}

export interface FileDeleteRequest {
  filepath: string;
}

export interface FileDeleteResponse {
  success: boolean;
  message: string;
}

export async function handleFileUpload(
  req: FileUploadRequest,
  userId: string | null = null
): Promise<FileUploadResponse> {
  const fs = require("fs");
  const path = require("path");
  const { config } = require("../config");
  const { getUserDataPath, ensureUserDirectory } = require("../utils/userContext");
  
  try {
    const baseDir = path.join(config.PROJECT_ROOT, "files");
    const filesDir = getUserDataPath(baseDir, userId);
    ensureUserDirectory(filesDir);
    
    const filepath = path.join(filesDir, req.filename);
    fs.writeFileSync(filepath, req.content, "utf8");
    
    const relativePath = userId ? `files/${userId}/${req.filename}` : `files/${req.filename}`;
    return {
      success: true,
      message: "File uploaded successfully",
      filepath: relativePath,
    };
  } catch (error: any) {
    return {
      success: false,
      message: error.message || "Failed to upload file",
      filepath: "",
    };
  }
}

export async function handleFileDelete(
  req: FileDeleteRequest,
  userId: string | null = null
): Promise<FileDeleteResponse> {
  const fs = require("fs");
  const path = require("path");
  const { config } = require("../config");
  const { getUserDataPath } = require("../utils/userContext");
  
  try {
    const baseDir = path.join(config.PROJECT_ROOT, "files");
    const filesDir = getUserDataPath(baseDir, userId);
    const filepath = path.join(filesDir, req.filepath.replace(/^files\//, ""));
    
    // Make sure it's in the user's files directory for safety
    if (!filepath.startsWith(filesDir)) {
      return {
        success: false,
        message: "Can only delete files in your files directory",
      };
    }
    
    if (fs.existsSync(filepath)) {
      fs.unlinkSync(filepath);
      return {
        success: true,
        message: "File deleted successfully",
      };
    } else {
      return {
        success: false,
        message: "File not found",
      };
    }
  } catch (error: any) {
    return {
      success: false,
      message: error.message || "Failed to delete file",
    };
  }
}