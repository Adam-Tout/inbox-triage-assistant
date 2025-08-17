# Inbox Triage Assistant - Deployment Guide

This guide covers deploying the Inbox Triage Assistant to various cloud platforms.

## Prerequisites

1. **Gmail API Setup**: Ensure you have:
   - Gmail API enabled in Google Cloud Console
   - OAuth 2.0 credentials downloaded as `credentials.json`
   - Added your email as a test user in OAuth consent screen

2. **Environment Variables**: Set these in your cloud platform:
   - `SECRET_KEY`: A secure random string for Flask sessions
   - `FLASK_ENV`: Set to `production` for cloud deployment

## Deployment Options

### 1. Heroku Deployment

#### Quick Deploy (Recommended)
[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/yourusername/inbox-triage-assistant)

#### Manual Deploy
```bash
# Install Heroku CLI
# Create new Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set SECRET_KEY=your-secure-secret-key
heroku config:set FLASK_ENV=production

# Deploy
git push heroku main
```

### 2. Docker Deployment

#### Local Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t inbox-triage .
docker run -p 5000:5000 -e SECRET_KEY=your-secret-key inbox-triage
```

#### Cloud Docker (AWS ECS, Google Cloud Run, Azure Container Instances)
```bash
# Build image
docker build -t inbox-triage .

# Tag for your registry
docker tag inbox-triage your-registry/inbox-triage:latest

# Push to registry
docker push your-registry/inbox-triage:latest

# Deploy to your cloud platform
```

### 3. Railway Deployment

1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push

### 4. Render Deployment

1. Connect your GitHub repository to Render
2. Choose "Web Service"
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app`
5. Set environment variables

### 5. Google Cloud Platform

#### App Engine
```bash
# Create app.yaml
runtime: python310
entrypoint: gunicorn -b :$PORT app:app

# Deploy
gcloud app deploy
```

#### Cloud Run
```bash
# Build and deploy
gcloud run deploy inbox-triage \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Environment Variables

Set these in your cloud platform:

```bash
SECRET_KEY=your-secure-random-string
FLASK_ENV=production
PORT=5000  # Usually set automatically by cloud platform
```

## Security Considerations

1. **Never commit credentials**: Ensure `credentials.json` and `token.pickle` are in `.gitignore`
2. **Use environment variables**: Store sensitive data in environment variables
3. **HTTPS**: Most cloud platforms provide HTTPS automatically
4. **Rate limiting**: Consider adding rate limiting for production use

## Monitoring

- **Health Check**: Available at `/health` endpoint
- **Logs**: Monitor application logs in your cloud platform
- **Metrics**: Set up monitoring for response times and error rates

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Ensure OAuth consent screen is configured correctly
2. **Memory Issues**: scikit-learn can be memory-intensive; consider upgrading dyno/instance
3. **Timeout Issues**: Email processing can take time; increase timeout limits

### Debug Mode

For local debugging, set:
```bash
FLASK_ENV=development
```

## Scaling

- **Horizontal Scaling**: Most cloud platforms support auto-scaling
- **Vertical Scaling**: Upgrade instance size for better performance
- **Caching**: Consider adding Redis for session storage in production

## Cost Optimization

- **Free Tiers**: Heroku, Railway, and Render offer free tiers
- **Spot Instances**: Use spot instances on AWS for cost savings
- **Auto-scaling**: Scale down during low usage periods
