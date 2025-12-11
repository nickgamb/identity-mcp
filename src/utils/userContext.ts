/**
 * User Context Utilities
 * 
 * Extracts and manages user context from OIDC tokens.
 * Supports both authenticated (multi-user) and anonymous (single-user) modes.
 */

import { Request } from "express";
import { config } from "../config";
import { logger } from "./logger";

export interface UserContext {
  userId: string | null;
  isAuthenticated: boolean;
  claims?: {
    sub?: string;
    email?: string;
    preferred_username?: string;
    [key: string]: any;
  };
}

/**
 * Extract user context from request.
 * Supports:
 * - OIDC tokens in Authorization header
 * - Anonymous access (when OIDC is disabled or not required)
 * 
 * IMPORTANT: If OIDC is enabled, userId is REQUIRED (not optional).
 * If OIDC is disabled, userId can be null (backward compatibility).
 */
export function getUserContext(req: Request): UserContext {
  // If OIDC is disabled, allow anonymous access (backward compatibility)
  if (!config.OIDC_ENABLED) {
    return {
      userId: null,
      isAuthenticated: false,
    };
  }

  // OIDC is enabled - userId is REQUIRED
  // The middleware should have validated and attached user context
  const userContext = (req as any).user as UserContext | undefined;
  if (userContext && userContext.isAuthenticated && userContext.userId) {
    return userContext;
  }

  // OIDC is enabled but no valid user context found
  // If auth is required, this will be rejected by middleware
  // If not required, we still return unauthenticated (but OIDC is enabled)
  return {
    userId: null,
    isAuthenticated: false,
  };
}

/**
 * Get userId from context, enforcing requirement when OIDC is enabled
 * @throws Error if OIDC is enabled but userId is null
 */
export function getRequiredUserId(userContext: UserContext | null): string | null {
  if (!config.OIDC_ENABLED) {
    // OIDC disabled - null is OK
    return userContext?.userId || null;
  }

  // OIDC enabled - userId is REQUIRED
  if (!userContext || !userContext.userId) {
    throw new Error("User ID is required when OIDC is enabled");
  }

  return userContext.userId;
}

/**
 * Get user-specific directory path.
 * Returns per-user path if authenticated, otherwise returns base path (backward compat).
 */
export function getUserDataPath(basePath: string, userId: string | null): string {
  if (!config.OIDC_ENABLED || !userId) {
    // Backward compatibility: return base path for anonymous/single-user mode
    return basePath;
  }

  // Multi-user mode: return per-user path
  const path = require("path");
  return path.join(basePath, userId);
}

/**
 * Ensure user directory exists
 */
export function ensureUserDirectory(userPath: string): void {
  const fs = require("fs");
  if (!fs.existsSync(userPath)) {
    fs.mkdirSync(userPath, { recursive: true });
    logger.info("Created user directory", { userPath });
  }
}

