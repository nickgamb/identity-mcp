/**
 * OIDC Authentication Context for Dashboard
 * 
 * Provides authentication state and methods for OIDC login/logout.
 * Supports optional authentication (backward compatible when OIDC is disabled).
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { UserManager, User } from 'oidc-client-ts';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  isOidcEnabled: boolean;
  login: () => Promise<void>;
  logout: () => Promise<void>;
  getAccessToken: () => string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// OIDC configuration from environment variables
const getOidcConfig = () => {
  const enabled = import.meta.env.VITE_OIDC_ENABLED === 'true';
  const issuer = import.meta.env.VITE_OIDC_ISSUER || 'http://localhost:8081';
  const clientId = import.meta.env.VITE_OIDC_CLIENT_ID || 'dashboard';
  const redirectUri = import.meta.env.VITE_OIDC_REDIRECT_URI || window.location.origin;

  return {
    enabled,
    issuer,
    clientId,
    redirectUri,
  };
};

let userManager: UserManager | null = null;

const initializeUserManager = () => {
  const config = getOidcConfig();
  
  if (!config.enabled) {
    return null;
  }

  if (!userManager) {
    userManager = new UserManager({
      authority: config.issuer,
      client_id: config.clientId,
      redirect_uri: config.redirectUri,
      response_type: 'code',
      scope: 'openid profile email',
      post_logout_redirect_uri: config.redirectUri,
      automaticSilentRenew: true,
      silent_redirect_uri: `${config.redirectUri}/silent-renew.html`,
    });
  }

  return userManager;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const config = getOidcConfig();

  useEffect(() => {
    if (!config.enabled) {
      // OIDC disabled - allow anonymous access (backward compatibility)
      setIsLoading(false);
      return;
    }

    const manager = initializeUserManager();
    if (!manager) {
      setIsLoading(false);
      return;
    }

    // Check if user is already logged in
    manager.getUser().then((user) => {
      setUser(user);
      setIsLoading(false);
    }).catch((error) => {
      console.error('Error getting user:', error);
      setIsLoading(false);
    });

    // Listen for user loaded events
    manager.events.addUserLoaded((user) => {
      setUser(user);
    });

    manager.events.addUserUnloaded(() => {
      setUser(null);
    });

    // Handle redirect after login
    manager.signinRedirectCallback().then((user) => {
      setUser(user);
    }).catch((error) => {
      // Not a redirect callback, ignore error
      if (error.message !== 'No state in response') {
        console.error('Error in signin redirect callback:', error);
      }
    });

    return () => {
      manager.events.removeUserLoaded(() => {});
      manager.events.removeUserUnloaded(() => {});
    };
  }, []);

  const login = async () => {
    if (!config.enabled) {
      console.warn('OIDC is not enabled');
      return;
    }

    const manager = initializeUserManager();
    if (!manager) {
      return;
    }

    try {
      await manager.signinRedirect();
    } catch (error) {
      console.error('Error during login:', error);
      throw error;
    }
  };

  const logout = async () => {
    if (!config.enabled) {
      console.warn('OIDC is not enabled');
      return;
    }

    const manager = initializeUserManager();
    if (!manager) {
      return;
    }

    try {
      await manager.signoutRedirect();
      setUser(null);
    } catch (error) {
      console.error('Error during logout:', error);
      throw error;
    }
  };

  const getAccessToken = (): string | null => {
    if (!user || !user.access_token) {
      return null;
    }
    return user.access_token;
  };

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated: !!user,
    isOidcEnabled: config.enabled,
    login,
    logout,
    getAccessToken,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

