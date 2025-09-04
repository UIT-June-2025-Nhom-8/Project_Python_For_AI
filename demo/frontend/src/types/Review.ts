export interface SentimentAnalysis {
  label: 'positive' | 'negative';
  confidence: number;
  probabilities: {
    positive: number;
    negative: number;
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
  };
}

export interface Review {
  id: string;
  text: string;
  timestamp: Date;
  sentiment: SentimentAnalysis;
  topics: TopicDetection;
}