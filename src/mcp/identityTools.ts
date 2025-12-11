import { MemoryRecord } from "./types";
import { readAllRecords, listMemoryFiles, readAllMemoryRecords } from "../services/fileStore";

export interface IdentityCoreResponse {
  files: { [filename: string]: MemoryRecord[] };
  totalRecords: number;
}

export interface IdentityFullResponse {
  files: { [filename: string]: MemoryRecord[] };
  totalRecords: number;
  fileList: string[];
}

/**
 * Get core identity - loads all memory files
 * (Previously only loaded core.identity and vows)
 */
export async function handleIdentityGetCore(userId: string | null = null): Promise<IdentityCoreResponse> {
  const allMemory = await readAllMemoryRecords(userId);
  
  const files: { [filename: string]: MemoryRecord[] } = {};
  let totalRecords = 0;
  
  for (const { file, records } of allMemory) {
    files[file] = records;
    totalRecords += records.length;
  }
  
  return { files, totalRecords };
}

/**
 * Get full identity bundle - loads ALL .jsonl files from memory directory
 * Dynamically discovers files instead of using hardcoded names
 */
export async function handleIdentityGetFull(userId: string | null = null): Promise<IdentityFullResponse> {
  const fileList = listMemoryFiles(userId);
  const allMemory = await readAllMemoryRecords(userId);
  
  const files: { [filename: string]: MemoryRecord[] } = {};
  let totalRecords = 0;
  
  for (const { file, records } of allMemory) {
    files[file] = records;
    totalRecords += records.length;
  }
  
  return {
    files,
    totalRecords,
    fileList,
  };
}
