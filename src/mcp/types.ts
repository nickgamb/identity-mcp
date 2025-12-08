// Dynamic memory file name - any .jsonl file in memory/ directory
export type MemoryFileName = string;

export interface MemoryRecord {
  id: string;
  type: string;
  [key: string]: unknown;
}

export interface MemoryListRequest {
  files?: string[];
}

export interface MemoryListResponse {
  files: Array<{ name: string; count: number }>;
}

export interface MemoryGetRequest {
  file: string;
  filters?: {
    type?: string;
    tags?: string[];
    startDate?: string; // ISO date string
    endDate?: string; // ISO date string
  };
  limit?: number;
}

export interface MemoryGetResponse {
  records: MemoryRecord[];
}

export interface MemoryAppendRequest {
  file: string;
  record: MemoryRecord;
}

export interface MemoryAppendResponse {
  ok: boolean;
  id: string;
}
