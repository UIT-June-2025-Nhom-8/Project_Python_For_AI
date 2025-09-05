import { ModelListResponse, ModelInfo } from '../types/Review';
import { makeAPIRequest, APIError } from './apiService';

// Enhanced error class for model service
export class ModelServiceError extends APIError {
  constructor(message: string, status?: number, isNetworkError?: boolean) {
    super(message, status, isNetworkError);
    this.name = 'ModelServiceError';
  }
}

// Get list of available models
export const getAvailableModels = async (): Promise<ModelListResponse> => {
  try {
    const data = await makeAPIRequest<ModelListResponse>('/models/list', {
      method: 'GET'
    });
    
    // Validate the response structure
    if (!data.sentiment_models || !data.topic_models) {
      throw new ModelServiceError('Invalid response format from models list API');
    }

    return data;
    
  } catch (error) {
    console.error('Error fetching available models:', error);
    
    if (error instanceof APIError) {
      throw new ModelServiceError(error.message, error.status, error.isNetworkError);
    }
    
    if (error instanceof Error) {
      throw new ModelServiceError(`Unexpected error: ${error.message}`, undefined, true);
    }
    
    throw new ModelServiceError('Unknown error occurred', undefined, true);
  }
};

// Switch the current active model
export const switchModel = async (modelType: 'sentiment' | 'topic', modelKey: string): Promise<{ status: string; message: string; current_model: string }> => {
  try {
    const data = await makeAPIRequest<{ status: string; message: string; current_model: string }>('/models/switch', {
      method: 'POST',
      body: JSON.stringify({
        model_type: modelType,
        model_key: modelKey
      })
    });
    
    if (!data.status || !data.message) {
      throw new ModelServiceError('Invalid response format from model switch API');
    }

    return data;
    
  } catch (error) {
    console.error('Error switching model:', error);
    
    if (error instanceof APIError) {
      throw new ModelServiceError(error.message, error.status, error.isNetworkError);
    }
    
    if (error instanceof Error) {
      throw new ModelServiceError(`Unexpected error: ${error.message}`, undefined, true);
    }
    
    throw new ModelServiceError('Unknown error occurred', undefined, true);
  }
};

// Get current model information
export const getModelInfo = async (): Promise<{
  sentiment_models_loaded: number;
  topic_models_loaded: number;
  current_sentiment_model: string;
  current_topic_model: string;
  sentiment_models: Record<string, string>;
  topic_models: Record<string, { name: string; num_topics: number }>;
}> => {
  try {
    const data = await makeAPIRequest<any>('/models/info', {
      method: 'GET'
    });
    
    return data;
    
  } catch (error) {
    console.error('Error fetching model info:', error);
    
    if (error instanceof APIError) {
      throw new ModelServiceError(error.message, error.status, error.isNetworkError);
    }
    
    if (error instanceof Error) {
      throw new ModelServiceError(`Unexpected error: ${error.message}`, undefined, true);
    }
    
    throw new ModelServiceError('Unknown error occurred', undefined, true);
  }
};
