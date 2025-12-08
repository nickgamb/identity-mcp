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
