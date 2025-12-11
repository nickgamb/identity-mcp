/**
 * API utility functions for making authenticated requests
 * Automatically includes OIDC tokens when available
 */

/**
 * Create fetch options with authentication token if available
 */
export function createFetchOptions(options: RequestInit = {}): RequestInit {
  // Get token from auth context (if available)
  // Note: This is a simple approach. In a real app, you might want to use
  // a more sophisticated method to get the token without React hooks.
  const token = getAuthToken();
  
  const headers = new Headers(options.headers);
  
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  
  return {
    ...options,
    headers,
  };
}

/**
 * Get auth token from localStorage (set by AuthContext)
 * This is a workaround since we can't use hooks in utility functions.
 * The AuthContext should set this when user logs in.
 */
function getAuthToken(): string | null {
  try {
    const userStr = localStorage.getItem('oidc.user');
    if (userStr) {
      const user = JSON.parse(userStr);
      return user.access_token || null;
    }
  } catch (error) {
    console.error('Error getting auth token:', error);
  }
  return null;
}

/**
 * Fetch wrapper that automatically includes auth token
 */
export async function authenticatedFetch(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const fetchOptions = createFetchOptions(options);
  return fetch(url, fetchOptions);
}

