# Railway Deployment Guide
## Medical Coding Intelligence Platform - Single Service Deployment

### 🚀 Quick Deployment

1. **Connect to Railway**
   - Visit [railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "Deploy from GitHub repo"
   - Select this repository

2. **Configure Environment Variables**
   ```bash
   # Required API Keys
   OPENAI_API_KEY=your_openai_api_key
   UMLS_API_KEY=your_umls_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   
   # MongoDB (use Railway addon or external)
   MONGO_URL=mongodb://localhost:27017
   DB_NAME=medical_coding
   
   # Pinecone Configuration
   PINECONE_INDEX_NAME=biobert
   PINECONE_ENVIRONMENT=us-east-1
   ```

3. **Add Redis Service**
   - In Railway dashboard, click "Add Service"
   - Select "Redis"
   - No configuration needed - Railway handles connection

4. **Deploy**
   - Railway will automatically build and deploy
   - Build process:
     1. Installs Node.js dependencies
     2. Builds React frontend
     3. Installs Python dependencies
     4. Copies frontend build to backend/static/
   - App available at your Railway URL

### 📋 Pre-Deployment Checklist

- [ ] OpenAI API key configured
- [ ] UMLS API account and key
- [ ] Pinecone account and index created
- [ ] All environment variables set
- [ ] GitHub repository connected

### 🔧 Architecture

```
Railway Service
├── FastAPI Backend (serves API + static files)
├── React Frontend (built to backend/static/)
├── Redis Database (Railway addon)
└── Environment Variables
```

### 💰 Cost Estimation

- **Railway Service**: $5/month
- **Redis Addon**: $2/month
- **Total**: ~$7/month

### 🔗 External Dependencies

- **OpenAI**: Pay-per-use (~$5-20/month)
- **Pinecone**: Free tier or $70/month
- **MongoDB**: Railway addon $5/month or Atlas free

### 🏗️ Build Process

Railway automatically runs:
1. `npm ci --production` (frontend dependencies)
2. `npm run build` (React build)
3. `pip install -r requirements.txt` (Python dependencies)
4. `cp -r frontend/build/* backend/static/` (copy static files)
5. `uvicorn backend.server:app` (start server)

### 🐛 Troubleshooting

#### Build Failures
- Check that all dependencies are listed in requirements.txt
- Verify Node.js version compatibility
- Check Railway build logs

#### API Errors
- Verify all environment variables are set
- Check API key validity
- Confirm external service connectivity

#### Performance Issues
- Monitor Railway metrics
- Check Redis connection
- Review ML model loading times

### 📚 Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4 |
| `UMLS_API_KEY` | Yes | UMLS Terminology Services API key |
| `PINECONE_API_KEY` | Yes | Pinecone vector database API key |
| `MONGO_URL` | Yes | MongoDB connection string |
| `REDIS_HOST` | Auto | Set by Railway Redis addon |
| `PORT` | Auto | Set by Railway (usually 3000) |

### 🚀 Production Deployment

1. **Setup Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial deployment setup"
   git remote add origin YOUR_GITHUB_URL
   git push -u origin main
   ```

2. **Railway Configuration**
   - Connect GitHub repository
   - Add environment variables
   - Add Redis service
   - Deploy

3. **Post-Deployment**
   - Test all API endpoints
   - Verify frontend loads correctly
   - Check medical API connections
   - Test ML model functionality

### 📊 Monitoring

Railway provides:
- Automatic health checks
- Resource usage metrics
- Build and deployment logs
- Performance monitoring

### 🔄 Updates

To update the application:
1. Push changes to GitHub
2. Railway automatically rebuilds and deploys
3. Zero-downtime deployments
4. Rollback capability if needed

### 🆘 Support

For deployment issues:
- Check Railway documentation
- Review build logs
- Contact Railway support
- Check GitHub issues for this project