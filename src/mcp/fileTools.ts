import { FileLoader, FileDocument } from "../services/fileLoader";

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

const fileLoader = new FileLoader();

export async function handleFileList(
  req: FileListRequest
): Promise<FileListResponse> {
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
  req: FileGetRequest
): Promise<FileGetResponse> {
  const file = await fileLoader.loadFile(req.filepath);
  return { file: file || null };
}

export async function handleFileSearch(
  req: FileSearchRequest
): Promise<FileSearchResponse> {
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
  req: FileGetNumberedRequest
): Promise<FileSearchResponse> {
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
  req: FileUploadRequest
): Promise<FileUploadResponse> {
  const fs = require("fs");
  const path = require("path");
  const config = require("../config").config;
  
  try {
    const filesDir = path.join(config.PROJECT_ROOT, "files");
    fs.mkdirSync(filesDir, { recursive: true });
    
    const filepath = path.join(filesDir, req.filename);
    fs.writeFileSync(filepath, req.content, "utf8");
    
    return {
      success: true,
      message: "File uploaded successfully",
      filepath: `files/${req.filename}`,
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
  req: FileDeleteRequest
): Promise<FileDeleteResponse> {
  const fs = require("fs");
  const path = require("path");
  const config = require("../config").config;
  
  try {
    const filepath = path.join(config.PROJECT_ROOT, req.filepath);
    
    // Make sure it's in the files directory for safety
    if (!filepath.includes(path.join(config.PROJECT_ROOT, "files"))) {
      return {
        success: false,
        message: "Can only delete files in the files directory",
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