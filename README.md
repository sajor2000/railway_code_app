# Medical Coding Intelligence Platform

## 🏥 AI-Powered Medical Coding Assistant

A comprehensive platform that combines cutting-edge AI with authoritative medical databases to deliver instant, accurate medical coding assistance.

### ✨ Key Features

- **🤖 AI-Powered Chat**: Natural language queries with GPT-4 responses
- **🔍 Multi-Source Search**: UMLS, RxNorm, ICD-10, SNOMED CT, LOINC integration
- **🧬 Hybrid RAG Search**: Combines API results with BioBERT semantic search
- **📊 Batch Processing**: CSV upload for bulk medical concept mapping
- **📥 Export Options**: CSV and HTML report generation
- **🔗 Related Concepts**: Semantic relationship exploration

## 🚀 Quick Start

### Google Cloud Run Deployment (Production Ready)

**Current Status**: ✅ Incremental deployment working, 🚀 Production ready with lazy loading

```bash
# Quick deploy to GCP
./deploy_production.sh
```

### Railway Deployment (Alternative)

1. **One-Click Deploy**
   [![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/deploy)

2. **Manual Setup**
   - Fork this repository
   - Connect to [Railway](https://railway.app)
   - Add environment variables (see `.env.railway.example`)
   - Deploy automatically

### Local Development

```bash
# Clone repository
git clone <your-repo-url>
cd medical_coding_app

# Backend setup
pip install -r requirements.txt
cd backend
python -m uvicorn server:app --reload

# Frontend setup (separate terminal)
cd frontend
npm install
npm start
```

## 📋 Environment Configuration

Copy `.env.railway.example` and configure:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key
UMLS_API_KEY=your_umls_api_key  
PINECONE_API_KEY=your_pinecone_api_key

# Database
MONGO_URL=your_mongodb_url
REDIS_HOST=your_redis_host
```

## 🏗️ Architecture

### Single-Service Design
- **FastAPI Backend**: Serves both API and static frontend files
- **React Frontend**: Built and served from backend/static/
- **ML Models**: BioBERT + Sentence Transformers for semantic search
- **Vector Database**: Pinecone for medical concept embeddings
- **Cache**: Redis for API response caching

### Technology Stack
- **Backend**: Python 3.10, FastAPI, PyTorch
- **Frontend**: React 19, Tailwind CSS
- **AI/ML**: OpenAI GPT-4, BioBERT, Sentence Transformers
- **Database**: MongoDB, Redis, Pinecone
- **Deployment**: Railway, Docker

## 🧪 Testing

The system has been comprehensively tested:

- ✅ Backend API tests (100% passing)
- ✅ Medical API integration verified
- ✅ AI chat functionality validated
- ✅ Hybrid search operational
- ✅ Export functions working

Test sepsis queries like:
- "What are the codes for sepsis?"
- "Show me ICD-10 codes for severe sepsis"
- "Find SNOMED codes for septic shock"

## 📖 API Documentation

Once deployed, visit `/docs` for interactive API documentation.

### Key Endpoints
- `POST /api/chat` - AI-powered medical chat
- `POST /api/search` - Medical terminology search
- `POST /api/csv/upload` - Batch processing
- `GET /api/abbreviations` - Medical abbreviations

## 💰 Cost Breakdown

### Railway Deployment (~$7/month)
- Railway Service: $5/month
- Redis addon: $2/month

### External APIs (variable)
- OpenAI GPT-4: ~$5-20/month (usage-based)
- Pinecone: Free tier or $70/month
- MongoDB Atlas: Free tier available

## 🔒 Security Features

- API key encryption and secure storage
- CORS protection
- Input validation and sanitization
- Rate limiting
- HTTPS enforcement

## 📚 Medical Data Sources

- **UMLS**: Unified Medical Language System
- **RxNorm**: Medication terminology
- **ICD-10-CM**: Diagnosis codes
- **SNOMED CT**: Clinical terminology
- **LOINC**: Laboratory observations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- Documentation: Check this README and RAILWAY_DEPLOYMENT.md
- Issues: Use GitHub Issues
- Deployment: See Railway deployment guide

## 🎯 Roadmap

- [ ] Mobile application
- [ ] Additional medical ontologies
- [ ] Real-time collaboration
- [ ] Voice interface
- [ ] Multi-language support

---

Built with ❤️ for healthcare professionals worldwide