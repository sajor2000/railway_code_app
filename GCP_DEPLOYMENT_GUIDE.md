# 🏥 Google Cloud Platform Deployment Guide
## Medical Coding Intelligence Platform

### 🎯 **Overview**
Deploy your React + FastAPI medical coding application to GCP Cloud Run with professional healthcare-grade infrastructure.

**What you'll get:**
- **URL**: `https://medical-coding-app-[hash].a.run.app`
- **Custom Domain**: Optional `https://medical-coding.yourdomain.com`
- **Auto-scaling**: Handles 0 to thousands of medical professionals
- **Cost**: ~$10-30/month depending on usage

---

## 🚀 **One-Time Setup**

### **1. Create GCP Project**
```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login and create project
gcloud auth login
gcloud projects create medical-coding-app-$(date +%s) --name="Medical Coding App"
gcloud config set project YOUR_PROJECT_ID
```

### **2. Enable Required APIs**
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

### **3. Store API Keys in Secret Manager**
```bash
# Store your API keys securely
echo "your-openai-api-key" | gcloud secrets create openai-api-key --data-file=-
echo "your-umls-api-key" | gcloud secrets create umls-api-key --data-file=-
echo "your-pinecone-api-key" | gcloud secrets create pinecone-api-key --data-file=-
echo "your-loinc-username" | gcloud secrets create loinc-username --data-file=-
echo "your-loinc-password" | gcloud secrets create loinc-password --data-file=-

# Grant Cloud Run access to secrets
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

---

## 🔧 **Deployment Methods**

### **Method 1: Automated Cloud Build (Recommended)**
```bash
# Deploy from your local machine
gcloud builds submit --config cloudbuild.yaml .

# Or set up GitHub integration for automatic deployments
gcloud builds triggers create github \
    --repo-name=railway_code_app \
    --repo-owner=sajor2000 \
    --branch-pattern="^main$" \
    --build-config=cloudbuild.yaml
```

### **Method 2: Manual Docker Deployment**
```bash
# Build and deploy manually
docker build -f Dockerfile.gcp -t gcr.io/YOUR_PROJECT_ID/medical-coding-app .
docker push gcr.io/YOUR_PROJECT_ID/medical-coding-app

gcloud run deploy medical-coding-app \
    --image gcr.io/YOUR_PROJECT_ID/medical-coding-app \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2 \
    --allow-unauthenticated
```

---

## 🌐 **Custom Domain Setup (Optional)**

### **1. Map Custom Domain**
```bash
# Add your domain to Cloud Run
gcloud run domain-mappings create \
    --service medical-coding-app \
    --domain medical-coding.yourdomain.com \
    --region us-central1
```

### **2. Update DNS Records**
Add the CNAME record shown in the output to your domain's DNS:
```
medical-coding.yourdomain.com → ghs.googlehosted.com
```

---

## 📊 **Configuration Details**

### **Resource Allocation**
- **Memory**: 4GB (handles BioBERT processing)
- **CPU**: 2 vCPUs (parallel medical API calls)
- **Timeout**: 60 minutes (for large batch processing)
- **Concurrency**: 100 requests per instance
- **Max Instances**: 10 (auto-scales based on demand)

### **Environment Variables**
Automatically configured via Secret Manager:
- `OPENAI_API_KEY` → Medical AI processing
- `UMLS_API_KEY` → Medical terminology access
- `PINECONE_API_KEY` → Vector database for BioBERT
- `LOINC_USERNAME/PASSWORD` → Laboratory data access

### **Health Checks**
- **Endpoint**: `/api/health`
- **Interval**: 30 seconds
- **Timeout**: 30 seconds
- **Failure Threshold**: 3 consecutive failures

---

## 💰 **Cost Estimation**

### **Light Usage** (~100 requests/day)
- **Cloud Run**: $5-10/month
- **Container Registry**: $1/month
- **Secret Manager**: $0.10/month
- **Total**: ~$6-11/month

### **Medium Usage** (~1,000 requests/day)
- **Cloud Run**: $15-25/month
- **Container Registry**: $2/month
- **Secret Manager**: $0.50/month
- **Total**: ~$17-27/month

### **Heavy Usage** (~10,000 requests/day)
- **Cloud Run**: $50-100/month
- **Container Registry**: $5/month
- **Secret Manager**: $2/month
- **Total**: ~$57-107/month

*Note: Plus external API costs (OpenAI, Pinecone)*

---

## 🔍 **Monitoring & Debugging**

### **View Logs**
```bash
# Real-time logs
gcloud run services logs tail medical-coding-app --region=us-central1

# Historical logs in Cloud Console
https://console.cloud.google.com/run/detail/us-central1/medical-coding-app
```

### **Health Check**
```bash
# Test your deployment
curl https://medical-coding-app-[hash].a.run.app/api/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "checks": {
    "api": "healthy",
    "cache": "healthy"
  }
}
```

### **Performance Monitoring**
- **Cloud Monitoring**: Automatic metrics for requests, latency, errors
- **Cloud Trace**: Detailed request tracing for medical data processing
- **Cloud Logging**: Comprehensive logs for debugging medical queries

---

## 🏥 **Healthcare Compliance Notes**

### **HIPAA Readiness**
- ✅ **Encryption**: Data encrypted in transit and at rest
- ✅ **Access Control**: IAM-based permissions
- ✅ **Audit Logs**: All access logged and monitored
- ✅ **Geographic Control**: Data stays in specified regions
- ⚠️ **BAA Required**: Sign Business Associate Agreement with Google

### **Security Features**
- **VPC**: Optional private networking
- **Identity-Aware Proxy**: Enterprise authentication
- **Cloud Armor**: DDoS protection and WAF
- **Certificate Manager**: Automatic SSL certificates

---

## 🚀 **Go Live Checklist**

- [ ] GCP project created and billing enabled
- [ ] APIs enabled (Cloud Run, Cloud Build, Secret Manager)
- [ ] API keys stored in Secret Manager
- [ ] Code deployed via Cloud Build
- [ ] Health endpoint responding
- [ ] Custom domain configured (optional)
- [ ] Monitoring dashboards set up
- [ ] Team access configured

**Your medical coding platform is now ready to serve healthcare professionals worldwide!** 🌍🏥

---

## 📞 **Support & Resources**

- **GCP Documentation**: https://cloud.google.com/run/docs
- **Medical AI Resources**: https://cloud.google.com/healthcare-api
- **Cost Calculator**: https://cloud.google.com/products/calculator
- **Status Page**: https://status.cloud.google.com/