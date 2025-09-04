# Review Analyzer Frontend

Modern React TypeScript frontend for AI-powered sentiment analysis and topic detection of product reviews.

## Features

- **Real-time Analysis**: Submit reviews and get instant AI analysis
- **Sentiment Visualization**: Interactive charts showing sentiment probabilities
- **Topic Detection**: Visual display of detected topics with keywords
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Fallback Support**: Graceful degradation when backend unavailable

## Technology Stack

- **React 19** with TypeScript
- **Modern CSS Grid & Flexbox**
- **Fetch API** for backend communication
- **Environment-based Configuration**

## API Integration

This frontend connects to AI model backend APIs:

### Production (Netlify)
- Uses Netlify Functions: `/.netlify/functions/analyze/sentiment`
- Automatic API discovery via environment variables

### Development
- Local backend: `http://localhost:5000`
- Set `REACT_APP_API_URL=http://localhost:5000` in `.env`

### Custom Backend
- Set `REACT_APP_API_URL=https://your-api.com` in environment

## Quick Start

### Development
```bash
# Install dependencies
npm install

# Start development server
npm start
```

Runs on `http://localhost:3000`

### Production Build
```bash
# Build for production
npm run build

# Test production build locally
npx serve -s build
```

## Environment Configuration

### `.env` (Development)
```env
REACT_APP_API_URL=http://localhost:5000
```

### `.env.production` (Production)
```env
REACT_APP_API_URL=/.netlify/functions
```

## Deployment

### Netlify (Recommended)

1. **Automatic Deployment:**
   ```bash
   # Build command: npm run build
   # Publish directory: build
   ```

2. **Manual Deployment:**
   ```bash
   npm run build
   # Upload build/ folder to Netlify
   ```

### Other Platforms

**Vercel:**
```bash
npm run build
vercel --prod
```

**Firebase Hosting:**
```bash
npm run build
firebase deploy
```

**GitHub Pages:**
```bash
npm run build
# Deploy build/ folder to gh-pages branch
```

## API Endpoints Used

### Sentiment Analysis
```
POST /.netlify/functions/analyze/sentiment
{
  "text": "Product review text..."
}
```

### Topic Detection
```
POST /.netlify/functions/analyze/topics
{
  "text": "Product review text..."
}
```

### Health Check
```
GET /.netlify/functions/health
```

## Component Structure

```
src/
├── components/
│   ├── ReviewInput.tsx     # Review submission form
│   └── ReviewDisplay.tsx   # Analysis results display
├── services/
│   ├── sentimentService.ts # Sentiment API client
│   └── topicService.ts     # Topic detection API client
├── types/
│   └── Review.ts          # TypeScript interfaces
└── App.tsx                # Main application component
```

## Styling

- **CSS Modules** for component-specific styles
- **Responsive Design** with mobile-first approach
- **Color-coded Results** for sentiment visualization
- **Modern UI Elements** with smooth animations

## Browser Support

- Chrome 88+
- Firefox 85+
- Safari 14+
- Edge 88+

## Performance

- **Code Splitting** for optimal loading
- **Lazy Loading** of components
- **Optimized Builds** with React Scripts
- **Caching** via service workers

## Error Handling

- **API Failures**: Automatic fallback to keyword-based analysis
- **Network Issues**: User-friendly error messages
- **Invalid Input**: Client-side validation
- **Loading States**: Progress indicators during analysis

## Self-Contained Design

This frontend is completely independent:
- ✅ No external file dependencies
- ✅ Environment-based API configuration
- ✅ Fallback functionality when backend unavailable
- ✅ Ready for independent deployment

## Development Notes

### Adding New Features
1. Create new components in `src/components/`
2. Add API services in `src/services/`
3. Update TypeScript types in `src/types/`
4. Test with both real and fallback APIs

### Customization
- Update colors in CSS files
- Modify analysis display in `ReviewDisplay.tsx`
- Add new input fields in `ReviewInput.tsx`
- Extend API integration in service files