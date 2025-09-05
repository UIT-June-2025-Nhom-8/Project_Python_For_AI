import { TopicDetection } from '../types/Review';
import { makeAPIRequest, APIError, checkBackendHealth } from './apiService';

// Enhanced error class for better error handling
export class TopicAnalysisError extends APIError {
  constructor(message: string, status?: number, isNetworkError?: boolean) {
    super(message, status, isNetworkError);
    this.name = 'TopicAnalysisError';
  }
}

// Real topic detection using the trained Gensim LDA model - backend only
export const detectTopics = async (text: string, modelKey?: string): Promise<TopicDetection> => {
  if (!text || text.trim().length === 0) {
    throw new TopicAnalysisError('Text is required for topic detection');
  }

  try {
    const requestBody: any = { text: text.trim() };
    if (modelKey) {
      requestBody.model = modelKey;
    }

    const data = await makeAPIRequest<any>('/analyze/topics', {
      method: 'POST',
      body: JSON.stringify(requestBody)
    });
    
    // Validate the response structure
    if (!data.topics || !Array.isArray(data.topics) || !data.dominant_topic) {
      throw new TopicAnalysisError('Invalid response format from topic detection API');
    }

    // Ensure the response matches our interface
    return {
      topics: data.topics.map((topic: any) => ({
        id: topic.id,
        name: topic.name,
        words: topic.words || [],
        probability: topic.probability || 0
      })),
      dominant_topic: {
        id: data.dominant_topic.id,
        name: data.dominant_topic.name,
        probability: data.dominant_topic.probability,
        words: data.dominant_topic.words
      },
      model_used: data.model_used
    };
    
  } catch (error) {
    console.error('Error calling topic detection API:', error);
    
    if (error instanceof APIError) {
      throw new TopicAnalysisError(error.message, error.status, error.isNetworkError);
    }
    
    if (error instanceof Error) {
      throw new TopicAnalysisError(`Unexpected error: ${error.message}`, undefined, true);
    }
    
    throw new TopicAnalysisError('Unknown error occurred', undefined, true);
  }
};

// Check if the backend API is available
export const checkTopicAPIHealth = async (): Promise<boolean> => {
  try {
    const healthResult = await checkBackendHealth();
    return healthResult.isHealthy;
  } catch {
    return false;
  }
};