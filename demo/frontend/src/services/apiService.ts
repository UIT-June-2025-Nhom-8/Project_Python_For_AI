// Common API configuration and utilities
const API_BASE_URL = process.env.REACT_APP_API_URL || '/.netlify/functions';

// API configuration
export const API_CONFIG = {
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  }
};

// Generic API error class
export class APIError extends Error {
  constructor(
    message: string, 
    public status?: number, 
    public isNetworkError?: boolean,
    public originalError?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// Check overall backend health
export const checkBackendHealth = async (): Promise<{
  isHealthy: boolean;
  message: string;
  details?: any;
}> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout for health check

    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: API_CONFIG.headers,
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      return {
        isHealthy: false,
        message: `Backend returned status ${response.status}`,
        details: { status: response.status, statusText: response.statusText }
      };
    }

    const data = await response.json();
    return {
      isHealthy: true,
      message: 'Backend is healthy',
      details: data
    };
    
  } catch (error) {
    console.error('Health check failed:', error);
    
    // Type guard to check if error is an Error object
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        return {
          isHealthy: false,
          message: 'Backend health check timed out'
        };
      }
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        return {
          isHealthy: false,
          message: 'Cannot connect to backend - network error'
        };
      }
      
      return {
        isHealthy: false,
        message: `Health check failed: ${error.message}`,
        details: error
      };
    }
    
    // Handle non-Error objects
    return {
      isHealthy: false,
      message: 'Health check failed: Unknown error',
      details: error
    };
  }
};

// Get model information from backend
export const getModelInfo = async (): Promise<{
  sentiment_model_loaded: boolean;
  lda_model_loaded: boolean;
  sentiment_classes: string[];
  lda_num_topics: number;
}> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(`${API_BASE_URL}/models/info`, {
      method: 'GET',
      headers: API_CONFIG.headers,
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new APIError(`Model info request failed with status ${response.status}`, response.status);
    }

    const data = await response.json();
    return data;
    
  } catch (error) {
    console.error('Model info request failed:', error);
    
    if (error instanceof APIError) {
      throw error;
    }
    
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new APIError('Model info request timed out', undefined, true);
      }
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new APIError('Network error - cannot connect to backend', undefined, true);
      }
      
      throw new APIError(`Unexpected error: ${error.message}`, undefined, true, error);
    }
    
    throw new APIError('Unknown error occurred', undefined, true, error);
  }
};

// Make a generic API request with error handling
export const makeAPIRequest = async <T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.timeout);

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        ...API_CONFIG.headers,
        ...options.headers
      },
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorMessage = `API request failed with status ${response.status}`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorMessage;
      } catch {
        // If response is not JSON, use status text
        errorMessage = response.statusText || errorMessage;
      }
      throw new APIError(errorMessage, response.status);
    }

    const data = await response.json();
    return data as T;
    
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new APIError('Request timed out', undefined, true);
      }
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new APIError('Network error - cannot connect to backend', undefined, true);
      }
      
      throw new APIError(`Unexpected error: ${error.message}`, undefined, true, error);
    }
    
    throw new APIError('Unknown error occurred', undefined, true, error);
  }
};
