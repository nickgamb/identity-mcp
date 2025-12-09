/**
 * Fine-tuning tools for MCP
 * Allows the model to trigger LoRA fine-tuning on accumulated conversations
 */

import { logger } from "../utils/logger";
import { exec } from "child_process";
import { promisify } from "util";
import { readFile, writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";

const execAsync = promisify(exec);

export interface FinetuneStartRequest {
  model_name?: string; // e.g., "gpt-oss:20b" - defaults to current model
  dataset_source?: "conversations" | "memories" | "files" | "all"; // What to use for training (default: "all")
  epochs?: number; // Training epochs (default: 3)
  learning_rate?: number; // Learning rate (default: 2e-5)
  output_name?: string; // Name for the fine-tuned adapter
}

export interface FinetuneStartResponse {
  success: boolean;
  job_id: string;
  message: string;
  estimated_time?: string;
}

export interface FinetuneStatusRequest {
  job_id: string;
}

export interface FinetuneStatusResponse {
  status: "pending" | "running" | "completed" | "failed";
  progress?: number; // 0-100
  message?: string;
  adapter_path?: string; // Path to the trained adapter
}

export interface FinetuneListRequest {
  // No parameters
}

export interface FinetuneListResponse {
  jobs: Array<{
    job_id: string;
    status: "pending" | "running" | "completed" | "failed";
    progress: number;
    message: string;
    adapter_path?: string;
  }>;
}

export interface FinetuneCancelRequest {
  job_id: string;
}

export interface FinetuneCancelResponse {
  success: boolean;
  message: string;
}

export interface FinetuneExportDatasetRequest {
  dataset_source?: "conversations" | "memories" | "files" | "all";
  output_path?: string;
}

export interface FinetuneExportDatasetResponse {
  success: boolean;
  output_path: string;
  example_count: number;
}

// Track active fine-tuning jobs
const activeJobs = new Map<string, {
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  message: string;
  adapter_path?: string;
  process?: any;
}>();

/**
 * Start a LoRA fine-tuning job
 */
export async function handleFinetuneStart(
  req: FinetuneStartRequest
): Promise<FinetuneStartResponse> {
  const jobId = `finetune-${Date.now()}`;
  const modelName = req.model_name || "gpt-oss:20b";
  const datasetSource = req.dataset_source || "all"; // Default to all sources
  const epochs = req.epochs ?? 3;
  const learningRate = req.learning_rate ?? 2e-5;
  const outputName = req.output_name || `lora-${modelName.replace(/[:/]/g, "-")}-${Date.now()}`;

  logger.info("Starting fine-tuning job", {
    jobId,
    modelName,
    datasetSource,
    epochs,
    learningRate,
    outputName,
  });

  // Initialize job tracking
  activeJobs.set(jobId, {
    status: "pending",
    progress: 0,
    message: "Preparing dataset...",
  });

  // Start fine-tuning process asynchronously
  startFinetuneProcess(jobId, {
    modelName,
    datasetSource,
    epochs,
    learningRate,
    outputName,
  }).catch((error) => {
    logger.error("Fine-tuning job failed", { jobId, error: String(error) });
    const job = activeJobs.get(jobId);
    if (job) {
      job.status = "failed";
      job.message = `Error: ${error.message}`;
    }
  });

  return {
    success: true,
    job_id: jobId,
    message: `Fine-tuning job started. Use finetune_status with job_id: "${jobId}" to check progress.`,
    estimated_time: `~${epochs * 2} hours (approximate)`,
  };
}

/**
 * Check status of a fine-tuning job
 */
export async function handleFinetuneStatus(
  req: FinetuneStatusRequest
): Promise<FinetuneStatusResponse> {
  const job = activeJobs.get(req.job_id);
  
  if (!job) {
    throw new Error(`Job not found: ${req.job_id}`);
  }

  return {
    status: job.status,
    progress: job.progress,
    message: job.message,
    adapter_path: job.adapter_path,
  };
}

export async function handleFinetuneList(
  req: FinetuneListRequest
): Promise<FinetuneListResponse> {
  const jobs = Array.from(activeJobs.entries()).map(([job_id, job]) => ({
    job_id,
    status: job.status,
    progress: job.progress,
    message: job.message,
    adapter_path: job.adapter_path,
  }));

  return { jobs };
}

export async function handleFinetuneCancel(
  req: FinetuneCancelRequest
): Promise<FinetuneCancelResponse> {
  const job = activeJobs.get(req.job_id);
  
  if (!job) {
    throw new Error(`Job not found: ${req.job_id}`);
  }

  if (job.status === "completed") {
    return {
      success: false,
      message: "Cannot cancel a completed job",
    };
  }

  if (job.status === "failed") {
    return {
      success: false,
      message: "Job has already failed",
    };
  }

  // Kill the process if running
  if (job.process && job.status === "running") {
    try {
      job.process.kill();
    } catch (error) {
      logger.warn("Error killing fine-tuning process", { job_id: req.job_id, error });
    }
  }

  job.status = "failed";
  job.message = "Cancelled by user";
  job.progress = 0;

  return {
    success: true,
    message: `Job ${req.job_id} cancelled successfully`,
  };
}

export async function handleFinetuneExportDataset(
  req: FinetuneExportDatasetRequest
): Promise<FinetuneExportDatasetResponse> {
  const datasetSource = req.dataset_source || "all";
  const outputPath = req.output_path || join(process.cwd(), "training_data", `training-dataset-${Date.now()}.jsonl`);

  try {
    const datasetPath = await exportComprehensiveTrainingData(
      datasetSource as "conversations" | "memories" | "files" | "all"
    );

    // Count examples in the dataset
    const { readFile } = await import("fs/promises");
    const content = await readFile(datasetPath, "utf8");
    const lines = content.split("\n").filter(line => line.trim().length > 0);
    const exampleCount = lines.length;

    return {
      success: true,
      output_path: datasetPath,
      example_count: exampleCount,
    };
  } catch (error) {
    logger.error("Error exporting dataset", error);
    throw error;
  }
}

/**
 * Background process to handle fine-tuning
 */
async function startFinetuneProcess(
  jobId: string,
  config: {
    modelName: string;
    datasetSource: string;
    epochs: number;
    learningRate: number;
    outputName: string;
  }
) {
  const job = activeJobs.get(jobId);
  if (!job) return;

  try {
    job.status = "running";
    job.progress = 10;
    job.message = "Exporting conversations for training...";

    // Step 1: Export all data sources to training format
    const datasetPath = await exportComprehensiveTrainingData(
      config.datasetSource as "conversations" | "memories" | "files" | "all"
    );
    
    job.progress = 30;
    job.message = "Preparing LoRA training script...";

    // Step 2: Create training script
    const scriptPath = await createTrainingScript(config, datasetPath);
    
    job.progress = 40;
    job.message = "Starting LoRA training (this will take several hours)...";

    // Step 3: Run training (this is a long-running process)
    const adapterPath = await runLoRATraining(scriptPath, config.outputName, (progress, message) => {
      if (job) {
        job.progress = 40 + Math.floor(progress * 0.5); // 40-90% for training
        job.message = message;
      }
    });

    job.progress = 95;
    job.message = "Finalizing adapter...";

    // Step 4: Convert adapter for Ollama (if needed)
    await finalizeAdapter(adapterPath, config.outputName);

    job.progress = 100;
    job.status = "completed";
    job.message = `Fine-tuning complete! Adapter saved. You can now load it in Ollama.`;
    job.adapter_path = adapterPath;

    logger.info("Fine-tuning job completed", { jobId, adapterPath });

  } catch (error) {
    job.status = "failed";
    job.message = `Error: ${error instanceof Error ? error.message : String(error)}`;
    logger.error("Fine-tuning process failed", { jobId, error });
    throw error;
  }
}

/**
 * Export comprehensive training data from conversations, files, and memory
 */
async function exportComprehensiveTrainingData(
  source: "conversations" | "memories" | "files" | "all"
): Promise<string> {
  const outputDir = join(process.cwd(), "training_data");
  await mkdir(outputDir, { recursive: true });

  const outputPath = join(outputDir, `training-dataset-${Date.now()}.jsonl`);

  const trainingExamples: Array<{ instruction: string; response: string }> = [];

  // 1. Load conversations if needed
  if (source === "conversations" || source === "all") {
    const { ConversationLoader } = await import("../services/conversationLoader");
    const loader = new ConversationLoader();
    const conversations = await loader.loadAllConversations();
    
    logger.info("Processing conversations for training", { 
      total: conversations.length
    });
    
    for (const conv of conversations) {
      // Convert conversation messages to instruction-response pairs
      for (let i = 0; i < conv.messages.length - 1; i++) {
        const userMsg = conv.messages[i];
        const assistantMsg = conv.messages[i + 1];
        
        if (userMsg.role === "user" && assistantMsg.role === "assistant") {
          // Create instruction-response pairs
          trainingExamples.push({
            instruction: userMsg.content,
            response: assistantMsg.content,
          });
        }
      }
    }
    
    logger.info("Added conversation examples", { count: trainingExamples.length });
  }

  // 2. Load files from RAG storage if needed
  if (source === "files" || source === "all") {
    const { FileLoader } = await import("../services/fileLoader");
    const fileLoader = new FileLoader();
    const files = await fileLoader.listFiles();
    
    logger.info("Processing files for training", { count: files.length });
    
    for (const filePath of files) {
      try {
        const file = await fileLoader.loadFile(filePath);
        if (file && file.content) {
          // Format files as instruction-response pairs
          // Instruction: "Read this transmission about [topic]"
          // Response: The file content
          const title = file.title || file.filename.replace(/\.(txt|md)$/, "");
          const instruction = `Read this transmission: ${title}`;
          
          trainingExamples.push({
            instruction,
            response: file.content,
          });
        }
      } catch (error) {
        logger.warn("Error loading file for training", { filePath, error });
      }
    }
    
    // Count file examples added in this section
    const conversationExamplesCount = source === "all" ? 
      (await (async () => {
        const { ConversationLoader } = await import("../services/conversationLoader");
        const loader = new ConversationLoader();
        const convs = await loader.loadAllConversations();
        let count = 0;
        for (const conv of convs) {
          for (let i = 0; i < conv.messages.length - 1; i++) {
            if (conv.messages[i].role === "user" && conv.messages[i + 1].role === "assistant") {
              count++;
            }
          }
        }
        return count;
      })()) : 0;
    const fileExamplesCount = trainingExamples.length - conversationExamplesCount;
    
    logger.info("Added file examples", { count: fileExamplesCount });
  }

  // 3. Load memory records if needed
  if (source === "memories" || source === "all") {
    const { readAllRecords } = await import("../services/fileStore");
    const { listMemoryFiles } = await import("../services/fileStore");
    
    const memoryFiles = listMemoryFiles();
    logger.info("Processing memory records for training", { files: memoryFiles.length });
    
    for (const memoryFile of memoryFiles) {
      try {
        const records = await readAllRecords(memoryFile);
        
        for (const record of records) {
          if (record.content) {
            // Format memories as instruction-response pairs
            // Use the memory type/context as instruction
            const instruction = record.type 
              ? `Recall ${record.type}: ${record.context || "memory"}` 
              : `Recall: ${record.context || "memory"}`;
            
            trainingExamples.push({
              instruction,
              response: typeof record.content === "string" ? record.content : JSON.stringify(record.content),
            });
          }
        }
      } catch (error) {
        logger.warn("Error loading memory file for training", { memoryFile, error });
      }
    }
    
    logger.info("Added memory examples", { count: trainingExamples.length });
  }

  // Shuffle examples for better training
  for (let i = trainingExamples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [trainingExamples[i], trainingExamples[j]] = [trainingExamples[j], trainingExamples[i]];
  }

  // Write to JSONL format
  const lines = trainingExamples.map(ex => JSON.stringify(ex)).join("\n");
  await writeFile(outputPath, lines + "\n");

  logger.info("Exported comprehensive training dataset", {
    path: outputPath,
    totalExamples: trainingExamples.length,
    source,
  });

  return outputPath;
}

/**
 * Create LoRA training script
 */
async function createTrainingScript(
  config: {
    modelName: string;
    epochs: number;
    learningRate: number;
    outputName: string;
  },
  datasetPath: string
): Promise<string> {
  const scriptsDir = join(process.cwd(), "scripts");
  const scriptPath = join(scriptsDir, `finetune_lora_${Date.now()}.py`);

  const scriptContent = `#!/usr/bin/env python3
"""
LoRA Fine-tuning Script
Auto-generated by MCP fine-tuning tool
"""

import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
MODEL_NAME = "${config.modelName.replace(/:/g, "/")}"
DATASET_PATH = "${datasetPath}"
OUTPUT_DIR = "${join(process.cwd(), "adapters", config.outputName)}"
EPOCHS = ${config.epochs}
LEARNING_RATE = ${config.learningRate}

# Load model and tokenizer
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Adjust based on model architecture
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
print(f"Loading dataset: {DATASET_PATH}")
dataset = load_dataset("json", data_files=DATASET_PATH)

def tokenize_function(examples):
    # Format: {"instruction": "...", "response": "..."}
    texts = []
    for inst, resp in zip(examples["instruction"], examples["response"]):
        text = f"### Instruction:\\n{inst}\\n\\n### Response:\\n{resp}"
        texts.append(text)
    return tokenizer(texts, truncation=True, max_length=4096, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,  # Smaller batch for memory efficiency
    gradient_accumulation_steps=8,  # Increase to maintain effective batch size
    learning_rate=LEARNING_RATE,
    fp16=True,  # Use if GPU supports it
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,  # Keep only last 3 checkpoints
    evaluation_strategy="no",
    warmup_steps=100,  # Warmup for better convergence
    lr_scheduler_type="cosine",  # Cosine learning rate schedule
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# Train
print("Starting training...")
trainer.train()

# Save
print(f"Saving adapter to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete!")
`;

  await writeFile(scriptPath, scriptContent);
  // Make executable
  await execAsync(`chmod +x "${scriptPath}"`);

  return scriptPath;
}

/**
 * Run LoRA training
 */
async function runLoRATraining(
  scriptPath: string,
  outputName: string,
  progressCallback: (progress: number, message: string) => void
): Promise<string> {
  const adaptersDir = join(process.cwd(), "adapters", outputName);
  await mkdir(adaptersDir, { recursive: true });

  // Run training script
  // This is a long-running process, so we'll run it in the background
  const { stdout, stderr } = await execAsync(`python3 "${scriptPath}"`, {
    maxBuffer: 1024 * 1024 * 10, // 10MB buffer
  });

  // Parse output for progress (simplified - real implementation would parse training logs)
  progressCallback(100, "Training complete");

  return adaptersDir;
}

/**
 * Finalize adapter for Ollama
 */
async function finalizeAdapter(adapterPath: string, outputName: string): Promise<void> {
  logger.info("Finalizing adapter for Ollama", { adapterPath, outputName });
  
  try {
    // Create a Modelfile that references the base model + LoRA adapter
    const modelfilePath = join(adapterPath, "Modelfile");
    const modelfileContent = `# Fine-tuned model: ${outputName}
# LoRA adapter applied to base model

FROM base_model
ADAPTER ./adapter_model

# Optional: Customize parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# Model card
TEMPLATE """{{ .System }}
{{ .Prompt }}"""
`;

    await writeFile(modelfilePath, modelfileContent, "utf8");
    logger.info("Created Modelfile", { modelfilePath });

    // Attempt to register with Ollama if it's available
    try {
      const { stdout, stderr } = await execAsync(
        `ollama create ${outputName} -f "${modelfilePath}"`,
        { cwd: adapterPath }
      );
      
      logger.info("Registered adapter with Ollama", { 
        outputName, 
        stdout: stdout.trim(),
        stderr: stderr.trim()
      });
    } catch (ollamaError: any) {
      // Ollama might not be installed or running - that's okay
      logger.warn("Could not register with Ollama (this is optional)", { 
        error: ollamaError.message,
        note: "You can manually register later with: ollama create " + outputName + " -f " + modelfilePath
      });
    }
  } catch (error: any) {
    logger.error("Error finalizing adapter", { error: error.message });
    // Don't throw - this is a non-critical step
  }
}

