import { SentimentAnalysis } from '../types/Review';
import { makeAPIRequest, APIError, checkBackendHealth } from './apiService';

// Enhanced error class for better error handling
export class SentimentAnalysisError extends APIError {
  constructor(message: string, status?: number, isNetworkError?: boolean) {
    super(message, status, isNetworkError);
    this.name = 'SentimentAnalysisError';
  }
}

// Real sentiment analysis using the trained logistic regression model - backend only
export const analyzeSentiment = async (text: string, modelKey?: string): Promise<SentimentAnalysis> => {
  if (!text || text.trim().length === 0) {
    throw new SentimentAnalysisError('Text is required for sentiment analysis');
  }

  try {
    const requestBody: any = { text: text.trim() };
    if (modelKey) {
      requestBody.model = modelKey;
    }

    const data = await makeAPIRequest<any>('/analyze/sentiment', {
      method: 'POST',
      body: JSON.stringify(requestBody)
    });
    
    // Validate the response structure
    if (!data.label || typeof data.confidence !== 'number' || !data.probabilities) {
      throw new SentimentAnalysisError('Invalid response format from sentiment analysis API');
    }

    // Ensure the response matches our interface
    return {
      label: data.label as 'positive' | 'negative',
      confidence: data.confidence,
      probabilities: {
        positive: data.probabilities.positive || 0,
        negative: data.probabilities.negative || 0
      },
      model_used: data.model_used
    };
    
  } catch (error) {
    console.error('Error calling sentiment analysis API:', error);
    
    if (error instanceof APIError) {
      throw new SentimentAnalysisError(error.message, error.status, error.isNetworkError);
    }
    
    if (error instanceof Error) {
      throw new SentimentAnalysisError(`Unexpected error: ${error.message}`, undefined, true);
    }
    
    throw new SentimentAnalysisError('Unknown error occurred', undefined, true);
  }
};

// Check if the backend API is available
export const checkSentimentAPIHealth = async (): Promise<boolean> => {
  try {
    const healthResult = await checkBackendHealth();
    return healthResult.isHealthy;
  } catch {
    return false;
  }
};