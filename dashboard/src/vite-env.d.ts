/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_OIDC_ENABLED?: string;
  readonly VITE_OIDC_ISSUER?: string;
  readonly VITE_OIDC_CLIENT_ID?: string;
  readonly VITE_OIDC_REDIRECT_URI?: string;
  /** Full URL to LibreChat for the Chat tab iframe (default: same host, port 3080). */
  readonly VITE_LIBRECHAT_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

