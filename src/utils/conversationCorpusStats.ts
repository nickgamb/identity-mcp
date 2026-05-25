import fs from "fs";
import path from "path";

/** Counts aligned with train_identity_model.load_conversations(). */
export function countConversationCorpus(conversationsDir: string): {
  conversation_files: number;
  conversations_with_messages: number;
  parse_errors: number;
} {
  if (!fs.existsSync(conversationsDir)) {
    return { conversation_files: 0, conversations_with_messages: 0, parse_errors: 0 };
  }

  let conversation_files = 0;
  let conversations_with_messages = 0;
  let parse_errors = 0;

  const names = fs
    .readdirSync(conversationsDir)
    .filter((f) => f.startsWith("conversation_") && f.endsWith(".jsonl"))
    .sort();

  for (const name of names) {
    conversation_files += 1;
    const filePath = path.join(conversationsDir, name);
    let messages = 0;
    try {
      const content = fs.readFileSync(filePath, "utf8");
      for (const line of content.split("\n")) {
        if (line.trim()) {
          JSON.parse(line);
          messages += 1;
        }
      }
    } catch {
      parse_errors += 1;
      continue;
    }
    if (messages > 0) {
      conversations_with_messages += 1;
    }
  }

  return { conversation_files, conversations_with_messages, parse_errors };
}
