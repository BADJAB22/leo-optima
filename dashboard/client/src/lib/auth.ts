/**
 * Simple authentication service for LEO Optima Dashboard
 */

export const getApiKey = (): string | null => {
  return localStorage.getItem('leo_api_key');
};

export const setApiKey = (key: string): void => {
  localStorage.setItem('leo_api_key', key);
};

export const clearApiKey = (): void => {
  localStorage.removeItem('leo_api_key');
};

export const isAuthenticated = (): boolean => {
  return !!getApiKey();
};

export const getApiBaseUrl = (): string => {
  // In development, proxy to 8000. In production, use the same host or env var.
  return import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
};
