import { appendRecord, listMemoryFiles, readAllRecords } from "../services/fileStore";
import {
  MemoryAppendRequest,
  MemoryAppendResponse,
  MemoryFileName,
  MemoryGetRequest,
  MemoryGetResponse,
  MemoryListRequest,
  MemoryListResponse,
  MemoryRecord,
} from "./types";

export async function handleMemoryList(req: MemoryListRequest, userId: string | null = null): Promise<MemoryListResponse> {
  const targetFiles: MemoryFileName[] = req.files && req.files.length > 0 ? req.files : listMemoryFiles(userId);

  const filesWithCounts = await Promise.all(
    targetFiles.map(async (file) => {
      const records = await readAllRecords(file, userId);
      return { name: file, count: records.length };
    }),
  );

  return { files: filesWithCounts };
}

export async function handleMemoryGet(req: MemoryGetRequest, userId: string | null = null): Promise<MemoryGetResponse> {
  const { file, filters, limit } = req;
  let records: MemoryRecord[] = await readAllRecords(file, userId);

  if (filters?.type) {
    records = records.filter((r) => r.type === filters.type);
  }

  if (filters?.tags && filters.tags.length > 0) {
    records = records.filter((r) => {
      const t = (r as any).tags;
      if (!Array.isArray(t)) return false;
      return filters.tags!.every((tag) => t.includes(tag));
    });
  }

  // Date range filtering
  if (filters?.startDate || filters?.endDate) {
    const startTime = filters.startDate ? new Date(filters.startDate).getTime() : 0;
    const endTime = filters.endDate ? new Date(filters.endDate).getTime() : Date.now();
    
    records = records.filter((r) => {
      const recordDate = (r as any).createdAt || (r as any).timestamp || (r as any).date;
      if (!recordDate) return false;
      
      const recordTime = new Date(recordDate).getTime();
      return recordTime >= startTime && recordTime <= endTime;
    });
  }

  if (typeof limit === "number" && limit > 0) {
    records = records.slice(0, limit);
  }

  return { records };
}

export async function handleMemoryAppend(
  req: MemoryAppendRequest,
  userId: string | null = null
): Promise<MemoryAppendResponse> {
  const { file, record } = req;
  const id = record.id || `rec-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const toStore: MemoryRecord = { ...record, id };
  await appendRecord(file, toStore, userId);

  return { ok: true, id };
}


