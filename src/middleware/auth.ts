/**
 * OIDC Authentication Middleware
 * 
 * Validates OIDC tokens and extracts user context.
 * Supports optional authentication (backward compatible with single-user mode).
 */

import { Request, Response, NextFunction } from "express";
import { jwtVerify, createRemoteJWKSet, JWTPayload } from "jose";
import { config } from "../config";
import { logger } from "../utils/logger";
import { UserContext } from "../utils/userContext";

// Cache JWKS for token validation
let jwks: ReturnType<typeof createRemoteJWKSet> | null = null;

/**
 * Initialize JWKS (JSON Web Key Set) for token validation
 */
async function initializeJWKS(): Promise<void> {
  if (!config.OIDC_ENABLED) {
    return;
  }

  try {
    const jwksUrl = new URL(".well-known/jwks.json", config.OIDC_ISSUER);
    jwks = createRemoteJWKSet(jwksUrl);
    logger.info("Initialized JWKS for OIDC token validation", { jwksUrl: jwksUrl.toString() });
  } catch (error) {
    logger.error("Failed to initialize JWKS", { error: String(error) });
    throw error;
  }
}

// Initialize JWKS on module load (if OIDC is enabled)
if (config.OIDC_ENABLED) {
  initializeJWKS().catch((err) => {
    logger.warn("JWKS initialization failed, will retry on first request", { error: String(err) });
  });
}

/**
 * OIDC Authentication Middleware
 * 
 * Validates JWT tokens from OIDC issuer.
 * If OIDC is disabled, allows all requests (backward compatibility).
 * If OIDC is enabled but not required, allows anonymous requests.
 */
export async function authenticateToken(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  // If OIDC is disabled, skip authentication (backward compatibility)
  if (!config.OIDC_ENABLED) {
    (req as any).user = {
      userId: null,
      isAuthenticated: false,
    } as UserContext;
    return next();
  }

  // Extract token from Authorization header
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    // If auth is required, reject the request
    if (config.OIDC_REQUIRE_AUTH) {
      res.status(401).json({
        error: "unauthorized",
        message: "Authentication required. Please provide a valid OIDC token.",
      });
      return;
    }
    // If auth is optional, allow anonymous access
    (req as any).user = {
      userId: null,
      isAuthenticated: false,
    } as UserContext;
    return next();
  }

  const token = authHeader.substring(7); // Remove "Bearer " prefix

  try {
    // Ensure JWKS is initialized
    if (!jwks) {
      await initializeJWKS();
      if (!jwks) {
        throw new Error("JWKS not initialized");
      }
    }

    // Verify token
    const { payload } = await jwtVerify(token, jwks, {
      issuer: config.OIDC_ISSUER,
      audience: config.OIDC_AUDIENCE,
    });

    // Extract user context from token claims
    const userId = payload.sub || payload.preferred_username || payload.email || null;
    if (!userId) {
      throw new Error("Token missing user identifier (sub, preferred_username, or email)");
    }

    const userContext: UserContext = {
      userId: userId as string,
      isAuthenticated: true,
      claims: payload as Record<string, any>,
    };

    // Attach user context to request
    (req as any).user = userContext;

    logger.debug("Token validated successfully", { userId, hasClaims: !!payload });
    next();
  } catch (error) {
    logger.warn("Token validation failed", { error: String(error) });
    
    // If auth is required, reject the request
    if (config.OIDC_REQUIRE_AUTH) {
      res.status(401).json({
        error: "unauthorized",
        message: "Invalid or expired token",
        details: error instanceof Error ? error.message : String(error),
      });
      return;
    }
    
    // If auth is optional, allow anonymous access
    (req as any).user = {
      userId: null,
      isAuthenticated: false,
    } as UserContext;
    next();
  }
}

/**
 * Optional authentication middleware
 * Always allows the request, but attaches user context if token is valid
 */
export async function optionalAuth(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  // Always allow, but try to extract user context if token is present
  await authenticateToken(req, res, next);
}

