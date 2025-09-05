export interface ModelInfo {
  key: string;
  name: string;
  type?: string;
  filename?: string;
  num_topics?: number;
}

export interface SentimentAnalysis {
  label: 'positive' | 'negative';
  confidence: number;
  probabilities: {
    positive: number;
    negative: number;
  };
  model_used?: {
    key: string;
    name: string;
  };
}

export interface TopicDetection {
  topics: Array<{
    id: number;
    name: string;
    words: string[];
    probability: number;
  }>;
  dominant_topic: {
    id: number;
    name: string;
    probability: number;
    words?: string[];
  };
  model_used?: {
    key: string;
    name: string;
    type: string;
  };
}

export interface ModelListResponse {
  sentiment_models: ModelInfo[];
  topic_models: ModelInfo[];
  current_sentiment_model: string;
  current_topic_model: string;
}

export interface Review {
  id: string;
  text: string;
  timestamp: Date;
  sentiment: SentimentAnalysis;
  topics: TopicDetection;
}