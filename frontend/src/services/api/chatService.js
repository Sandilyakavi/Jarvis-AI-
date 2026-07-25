import { API_CONFIG } from '../../config/api';

/**
 * Custom error class for API interaction errors.
 */
export class APIError extends Error {
  constructor(message, code, details = null) {
    super(message);
    this.name = 'APIError';
    this.code = code;
    this.details = details;
  }
}

/**
 * Service to handle all chat-related backend API interactions.
 */
export const chatService = {
  /**
   * Sends a list of messages to the backend chat API.
   * Supports timeout via AbortController.
   * 
   * @param {Array<{role: string, content: string}>} messages - Conversation history
   * @param {Object} [options] - Additional parameters (provider, model, temperature, signal)
   * @returns {Promise<{message: {role: string, content: string}, provider: string, model: string}>}
   */
  async sendMessage(messages, options = {}) {
    const { provider, model, temperature, signal: customSignal } = options;

    // Create an AbortController for handling request timeout if no custom signal is provided
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.timeoutMs);
    const signal = customSignal || controller.signal;

    try {
      const response = await fetch(API_CONFIG.endpoints.chat, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages,
          provider: provider || null,
          model: model || null,
          temperature: temperature !== undefined ? temperature : 0.7,
        }),
        signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch (e) {
          throw new APIError('Invalid JSON response received from neural core.', 'INVALID_RESPONSE');
        }

        // Check if the response follows the StandardErrorResponse schema
        if (errorData && errorData.success === false && errorData.error) {
          throw new APIError(
            errorData.message || 'AI request failed.',
            errorData.error.code || 'BACKEND_ERROR',
            errorData.error.details
          );
        }

        throw new APIError(`HTTP error: ${response.status}`, 'HTTP_ERROR');
      }

      const responseData = await response.json();
      
      // Conforms to StandardResponse[ChatResponseData]
      if (responseData && responseData.success && responseData.data) {
        return responseData.data;
      }

      throw new APIError('Malformed data structure from core system.', 'MALFORMED_DATA');

    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === 'AbortError') {
        throw new APIError('Response telemetry request timed out.', 'TIMEOUT');
      }

      if (error instanceof APIError) {
        throw error;
      }

      // Check if it's a connection / network error
      if (error.message && error.message.includes('Failed to fetch')) {
        throw new APIError('Neural core terminal is offline.', 'OFFLINE');
      }

      throw new APIError(error.message || 'Unknown network error.', 'NETWORK_ERROR');
    }
  },

  /**
   * Stub function structure showing future compatibility for Server-Sent Events / streaming.
   * No actual implementation for Sprint 1 Day 3.
   */
  async streamMessage(messages, onChunk, onError, options = {}) {
    throw new Error('Streaming protocol is currently classified/unsupported.');
  }
};
