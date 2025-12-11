/**
 * Identity Verification Tools
 * 
 * Uses the trained identity embedding model to verify if messages
 * match the enrolled identity profile.
 * 
 * When the Python identity service is running, uses full semantic
 * verification. Falls back to stylistic-only if service unavailable.
 */

import fs from "fs";
import path from "path";
import { config } from "../config";
import { logger } from "../utils/logger";
import { getUserDataPath, ensureUserDirectory } from "../utils/userContext";

// Identity service configuration
let serviceAvailable: boolean | null = null; // null = not checked yet

interface IdentityConfig {
  model_name: string;
  model_size: string;
  num_messages: number;
  num_conversations: number;
  created_at: string;
  statistics: {
    num_samples: number;
    embedding_dim: number;
    mean_similarity: number;
    std_similarity: number;
    similarity_threshold_1std: number;
    similarity_threshold_2std: number;
  };
}

interface StylisticProfile {
  [feature: string]: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
}

interface VocabularyProfile {
  word_frequencies: Record<string, number>;
  bigrams: Record<string, number>;
  distinctive_terms: Record<string, number>;
  vocabulary_size: number;
  total_words: number;
}

interface VerificationResult {
  available: boolean;
  verified?: boolean;
  confidence?: "high" | "medium" | "low" | "none";
  scores?: {
    stylistic_match: number;
    vocabulary_match: number;
    combined_score: number;
  };
  details?: {
    stylistic_features: Record<string, { value: number; expected: number; match: number }>;
    vocabulary_overlap: number;
    distinctive_terms_found: string[];
  };
  thresholds?: {
    high_confidence: number;
    medium_confidence: number;
  };
  message?: string;
}

let cachedConfig: IdentityConfig | null = null;
let cachedStylisticProfile: StylisticProfile | null = null;
let cachedVocabularyProfile: VocabularyProfile | null = null;

// Python service response types
interface SemanticVerificationResult {
  success: boolean;
  similarity?: number;
  confidence?: string;
  threshold?: number;
  message?: string;
}

/**
 * Check if the Python identity service is available
 */
async function checkServiceAvailable(): Promise<boolean> {
  try {
    const response = await fetch(`${config.IDENTITY_SERVICE_URL}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(2000) // 2 second timeout
    });
    serviceAvailable = response.ok;
    return serviceAvailable;
  } catch {
    serviceAvailable = false;
    return false;
  }
}

/**
 * Call Python service for semantic verification
 */
async function callSemanticVerification(message: string, userId: string | null = null): Promise<SemanticVerificationResult | null> {
  // Check service availability (cache result for 30 seconds)
  if (serviceAvailable === null) {
    await checkServiceAvailable();
  }

  if (!serviceAvailable) {
    return null;
  }

  try {
    const response = await fetch(`${config.IDENTITY_SERVICE_URL}/verify`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        ...(userId ? { "X-User-Id": userId } : {})
      },
      body: JSON.stringify({ 
        message,
        ...(userId ? { user_id: userId } : {})
      }),
      signal: AbortSignal.timeout(10000) // 10 second timeout
    });

    if (!response.ok) {
      logger.warn("Python identity service returned error", { status: response.status });
      return null;
    }

    return await response.json() as SemanticVerificationResult;
  } catch (error) {
    logger.warn("Failed to call Python identity service", { error: String(error) });
    // Mark service as unavailable for next calls
    serviceAvailable = false;
    return null;
  }
}

/**
 * Reset service availability check (useful after service restart)
 */
export function resetServiceAvailability() {
  serviceAvailable = null;
}

function getModelsDir(userId: string | null = null): string {
  const baseDir = path.join(config.PROJECT_ROOT || process.cwd(), "models", "identity");
  const userDir = getUserDataPath(baseDir, userId);
  ensureUserDirectory(userDir);
  return userDir;
}

function loadIdentityConfig(userId: string | null = null): IdentityConfig | null {
  // Note: Cache is per-process, not per-user. For multi-user, we'd need per-user cache.
  // For now, we'll load fresh each time if userId is provided (multi-user mode)
  if (!userId && cachedConfig) return cachedConfig;

  const configPath = path.join(getModelsDir(userId), "config.json");
  if (!fs.existsSync(configPath)) {
    return null;
  }

  try {
    const content = fs.readFileSync(configPath, "utf8");
    cachedConfig = JSON.parse(content);
    return cachedConfig;
  } catch (error) {
    logger.error("Failed to load identity config", { error: String(error) });
    return null;
  }
}

function loadStylisticProfile(userId: string | null = null): StylisticProfile | null {
  if (!userId && cachedStylisticProfile) return cachedStylisticProfile;

  const profilePath = path.join(getModelsDir(userId), "stylistic_profile.json");
  if (!fs.existsSync(profilePath)) {
    return null;
  }

  try {
    const content = fs.readFileSync(profilePath, "utf8");
    cachedStylisticProfile = JSON.parse(content);
    return cachedStylisticProfile;
  } catch (error) {
    logger.error("Failed to load stylistic profile", { error: String(error) });
    return null;
  }
}

function loadVocabularyProfile(userId: string | null = null): VocabularyProfile | null {
  if (!userId && cachedVocabularyProfile) return cachedVocabularyProfile;

  const profilePath = path.join(getModelsDir(userId), "vocabulary_profile.json");
  if (!fs.existsSync(profilePath)) {
    return null;
  }

  try {
    const content = fs.readFileSync(profilePath, "utf8");
    cachedVocabularyProfile = JSON.parse(content);
    return cachedVocabularyProfile;
  } catch (error) {
    logger.error("Failed to load vocabulary profile", { error: String(error) });
    return null;
  }
}

function loadTemporalAnalysis(userId: string | null = null): any | null {
  const analysisPath = path.join(getModelsDir(userId), "temporal_analysis.json");
  if (!fs.existsSync(analysisPath)) {
    return null;
  }

  try {
    const content = fs.readFileSync(analysisPath, "utf8");
    return JSON.parse(content);
  } catch (error) {
    logger.error("Failed to load temporal analysis", { error: String(error) });
    return null;
  }
}

/**
 * Compute stylistic features from text (mirrors Python implementation)
 */
function computeStylisticFeatures(text: string): Record<string, number> {
  if (!text) return {};

  const charCount = text.length;
  const words = text.split(/\s+/);
  const wordCount = words.length;
  const sentences = text.split(/[.!?]+/).filter(s => s.trim());
  const sentenceCount = Math.max(sentences.length, 1);

  return {
    // Length metrics
    avg_word_length: words.reduce((sum, w) => sum + w.length, 0) / Math.max(wordCount, 1),
    avg_sentence_length: wordCount / sentenceCount,
    char_count: charCount,
    word_count: wordCount,

    // Punctuation density (per 100 chars)
    question_marks: (text.match(/\?/g) || []).length / charCount * 100,
    exclamation_marks: (text.match(/!/g) || []).length / charCount * 100,
    commas: (text.match(/,/g) || []).length / charCount * 100,
    periods: (text.match(/\./g) || []).length / charCount * 100,
    ellipsis: (text.match(/\.{3}|…/g) || []).length / charCount * 100,
    em_dashes: (text.match(/—|--/g) || []).length / charCount * 100,
    parentheses: (text.match(/\([^)]*\)/g) || []).length / charCount * 100,
    quotes: (text.match(/['"]/g) || []).length / charCount * 100,

    // Capitalization
    caps_ratio: (text.match(/[A-Z]/g) || []).length / Math.max(charCount, 1),
    all_caps_words: (text.match(/\b[A-Z]{2,}\b/g) || []).length / Math.max(wordCount, 1),

    // Structure
    newline_ratio: (text.match(/\n/g) || []).length / Math.max(charCount, 1),
    code_blocks: (text.match(/```/g) || []).length,

    // Linguistic markers
    first_person_singular: (text.match(/\b(I|me|my|mine|myself)\b/gi) || []).length / Math.max(wordCount, 1),
    first_person_plural: (text.match(/\b(we|us|our|ours|ourselves)\b/gi) || []).length / Math.max(wordCount, 1),
    hedging_words: (text.match(/\b(maybe|perhaps|possibly|might|could|seems?|think|believe|feel)\b/gi) || []).length / Math.max(wordCount, 1),
    certainty_words: (text.match(/\b(definitely|certainly|absolutely|clearly|obviously|always|never)\b/gi) || []).length / Math.max(wordCount, 1),
  };
}

/**
 * Compute vocabulary overlap with profile
 */
function computeVocabularyMatch(
  text: string,
  profile: VocabularyProfile
): { overlap: number; distinctiveFound: string[] } {
  const words = text.toLowerCase().match(/\b[a-z]{3,}\b/g) || [];
  const wordSet = new Set(words);

  // Check overlap with frequent words
  const profileWords = Object.keys(profile.word_frequencies);
  const overlap = profileWords.filter(w => wordSet.has(w)).length / Math.max(profileWords.length, 1);

  // Check for distinctive terms
  const distinctiveFound = Object.keys(profile.distinctive_terms).filter(term => 
    text.toLowerCase().includes(term)
  );

  return { overlap, distinctiveFound };
}

/**
 * Get identity model status
 */
export async function handleIdentityModelStatus(userId: string | null = null): Promise<{
  exists: boolean;
  available?: boolean;
  config?: IdentityConfig;
  stylistic_profile?: StylisticProfile;
  vocabulary_profile?: VocabularyProfile;
  temporal_analysis?: any;
  identity_report?: string;
  message?: string;
}> {
  const identityConfig = loadIdentityConfig(userId);
  const stylisticProfile = loadStylisticProfile(userId);
  const vocabularyProfile = loadVocabularyProfile(userId);
  const temporalAnalysis = loadTemporalAnalysis(userId);

  if (!identityConfig) {
    return {
      exists: false,
      available: false,
      message: "Identity model not found. Run: python scripts/identity_model/train_identity_model.py"
    };
  }

  // Try to load identity report from memory directory (per-user)
  let identityReport: string | undefined = undefined;
  try {
    const { getUserDataPath } = require("../utils/userContext");
    const memoryDir = getUserDataPath(config.MEMORY_DIR, userId);
    const reportPath = path.join(memoryDir, "identity_report.md");
    if (fs.existsSync(reportPath)) {
      identityReport = fs.readFileSync(reportPath, "utf8");
    }
  } catch (error) {
    logger.warn("Failed to load identity report", { error: String(error) });
  }

  return {
    exists: true,
    available: true,
    config: identityConfig,
    stylistic_profile: stylisticProfile || undefined,
    vocabulary_profile: vocabularyProfile || undefined,
    temporal_analysis: temporalAnalysis || undefined,
    identity_report: identityReport
  };
}

/**
 * Verify a message against the identity profile
 * 
 * Uses semantic verification via Python service when available (most accurate).
 * Falls back to stylistic + vocabulary verification if service unavailable.
 */
export async function handleIdentityVerify({ 
  message 
}: { 
  message: string 
}, userId: string | null = null): Promise<VerificationResult & { semantic_available?: boolean; semantic_score?: number }> {
  const identityConfig = loadIdentityConfig(userId);
  const stylisticProfile = loadStylisticProfile(userId);
  const vocabularyProfile = loadVocabularyProfile(userId);

  if (!identityConfig || !stylisticProfile || !vocabularyProfile) {
    return {
      available: false,
      message: "Identity model not found. Run: python scripts/identity_model/train_identity_model.py"
    };
  }

  // Try semantic verification first (most accurate)
  const semanticResult = await callSemanticVerification(message, userId);
  
  // Compute stylistic features (always do this for detailed breakdown)
  const msgFeatures = computeStylisticFeatures(message);
  
  // Compare to profile
  const featureMatches: Record<string, { value: number; expected: number; match: number }> = {};
  const matchScores: number[] = [];

  for (const [feature, profile] of Object.entries(stylisticProfile)) {
    if (feature in msgFeatures) {
      const value = msgFeatures[feature];
      const expected = profile.mean;
      const std = profile.std + 0.0001; // Avoid division by zero
      
      // Z-score based match (higher = better match)
      const zScore = Math.abs(value - expected) / std;
      const match = Math.max(0, 1 - zScore / 3); // 3 std = 0 match
      
      featureMatches[feature] = { value, expected, match };
      matchScores.push(match);
    }
  }

  const stylisticMatch = matchScores.length > 0 
    ? matchScores.reduce((a, b) => a + b, 0) / matchScores.length 
    : 0.5;

  // Compute vocabulary match
  const { overlap, distinctiveFound } = computeVocabularyMatch(message, vocabularyProfile);
  const vocabularyMatch = (overlap + (distinctiveFound.length > 0 ? 0.3 : 0)) / 1.3;

  // Determine final score - use semantic if available, otherwise stylistic+vocab
  let combinedScore: number;
  let semanticAvailable = false;
  let semanticScore: number | undefined;

  if (semanticResult && semanticResult.success && semanticResult.similarity !== undefined) {
    // Semantic available: weight it heavily (60% semantic, 25% stylistic, 15% vocabulary)
    semanticScore = semanticResult.similarity;
    semanticAvailable = true;
    combinedScore = 0.6 * semanticScore + 0.25 * stylisticMatch + 0.15 * vocabularyMatch;
    logger.info("Identity verification using semantic model", { 
      semantic: semanticScore, 
      stylistic: stylisticMatch,
      vocabulary: vocabularyMatch,
      combined: combinedScore 
    });
  } else {
    // No semantic: fallback to stylistic + vocabulary only
    combinedScore = 0.6 * stylisticMatch + 0.4 * vocabularyMatch;
    logger.info("Identity verification using stylistic/vocabulary only", { 
      stylistic: stylisticMatch,
      vocabulary: vocabularyMatch,
      combined: combinedScore 
    });
  }

  // Determine confidence
  let confidence: "high" | "medium" | "low" | "none";
  let verified: boolean;

  if (combinedScore >= 0.7) {
    confidence = "high";
    verified = true;
  } else if (combinedScore >= 0.5) {
    confidence = "medium";
    verified = true;
  } else if (combinedScore >= 0.3) {
    confidence = "low";
    verified = false;
  } else {
    confidence = "none";
    verified = false;
  }

  return {
    available: true,
    verified,
    confidence,
    semantic_available: semanticAvailable,
    semantic_score: semanticScore,
    scores: {
      stylistic_match: stylisticMatch,
      vocabulary_match: vocabularyMatch,
      combined_score: combinedScore
    },
    details: {
      stylistic_features: featureMatches,
      vocabulary_overlap: overlap,
      distinctive_terms_found: distinctiveFound
    },
    thresholds: {
      high_confidence: 0.7,
      medium_confidence: 0.5
    },
    message: semanticAvailable 
      ? "Verified using semantic + stylistic analysis" 
      : "Verified using stylistic analysis only (start Python identity service for full semantic verification)"
  };
}

/**
 * Verify a conversation (multiple messages)
 */
export async function handleIdentityVerifyConversation({
  messages
}: {
  messages: string[]
}, userId: string | null = null): Promise<{
  available: boolean;
  overall_verified?: boolean;
  overall_confidence?: "high" | "medium" | "low" | "none";
  overall_score?: number;
  message_results?: Array<{
    message_preview: string;
    verified: boolean;
    confidence: string;
    score: number;
  }>;
  message?: string;
}> {
  if (!messages || messages.length === 0) {
    return {
      available: false,
      message: "No messages provided"
    };
  }

  const results = [];
  let totalScore = 0;

  for (const msg of messages) {
    const result = await handleIdentityVerify({ message: msg }, userId);
    
    if (!result.available) {
      return {
        available: false,
        message: result.message
      };
    }

    results.push({
      message_preview: msg.substring(0, 50) + (msg.length > 50 ? "..." : ""),
      verified: result.verified || false,
      confidence: result.confidence || "none",
      score: result.scores?.combined_score || 0
    });

    totalScore += result.scores?.combined_score || 0;
  }

  const overallScore = totalScore / messages.length;
  
  let overallConfidence: "high" | "medium" | "low" | "none";
  let overallVerified: boolean;

  if (overallScore >= 0.7) {
    overallConfidence = "high";
    overallVerified = true;
  } else if (overallScore >= 0.5) {
    overallConfidence = "medium";
    overallVerified = true;
  } else if (overallScore >= 0.3) {
    overallConfidence = "low";
    overallVerified = false;
  } else {
    overallConfidence = "none";
    overallVerified = false;
  }

  return {
    available: true,
    overall_verified: overallVerified,
    overall_confidence: overallConfidence,
    overall_score: overallScore,
    message_results: results
  };
}

/**
 * Get identity profile summary (for debugging/inspection)
 */
export async function handleIdentityProfileSummary(userId: string | null = null): Promise<{
  available: boolean;
  stylistic_summary?: Record<string, { mean: number; std: number }>;
  vocabulary_summary?: {
    vocabulary_size: number;
    top_words: string[];
    distinctive_terms: string[];
  };
  training_info?: {
    messages_trained_on: number;
    conversations: number;
    created_at: string;
  };
  message?: string;
}> {
  const identityConfig = loadIdentityConfig(userId);
  const stylisticProfile = loadStylisticProfile(userId);
  const vocabularyProfile = loadVocabularyProfile(userId);

  if (!identityConfig || !stylisticProfile || !vocabularyProfile) {
    return {
      available: false,
      message: "Identity model not found. Run: python scripts/identity_model/train_identity_model.py"
    };
  }

  // Summarize stylistic profile
  const stylisticSummary: Record<string, { mean: number; std: number }> = {};
  for (const [feature, stats] of Object.entries(stylisticProfile)) {
    stylisticSummary[feature] = {
      mean: Math.round(stats.mean * 1000) / 1000,
      std: Math.round(stats.std * 1000) / 1000
    };
  }

  // Summarize vocabulary
  const topWords = Object.entries(vocabularyProfile.word_frequencies)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([word]) => word);

  const distinctiveTerms = Object.keys(vocabularyProfile.distinctive_terms).slice(0, 20);

  return {
    available: true,
    stylistic_summary: stylisticSummary,
    vocabulary_summary: {
      vocabulary_size: vocabularyProfile.vocabulary_size,
      top_words: topWords,
      distinctive_terms: distinctiveTerms
    },
    training_info: {
      messages_trained_on: identityConfig.num_messages,
      conversations: identityConfig.num_conversations,
      created_at: identityConfig.created_at
    }
  };
}

/**
 * Clear cached identity data (useful after retraining)
 */
export function clearIdentityVerificationCache() {
  cachedConfig = null;
  cachedStylisticProfile = null;
  cachedVocabularyProfile = null;
}

