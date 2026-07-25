// Centralized API Configuration
// Supports environment variables, development, and production settings.
// Do not hardcode URLs.

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const API_CONFIG = {
  baseUrl: API_BASE_URL,
  endpoints: {
    chat: `${API_BASE_URL}/api/chat`,
    health: `${API_BASE_URL}/health`,
  },
  timeoutMs: 15000, // 15 seconds request timeout
};
