#!/bin/bash
# Production V2 deployment - Ultra-fast startup

echo "🚀 Deploying Medical Coding Intelligence Platform - Production V2"
echo "⚡ This version has ultra-fast startup with complete lazy loading"

# Configuration
PROJECT_ID="medical-coding-intelligent"
SERVICE_NAME="medicalcodeassistant"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME-prod-v2"

# Build and push
echo "🔨 Building ultra-fast production image..."
docker build -f Dockerfile.production-v2 -t $IMAGE_NAME .

echo "📤 Pushing to Container Registry..."
docker push $IMAGE_NAME

# Deploy with optimized settings
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --set-env-vars "$(cat <<EOF
PORT=8080
OPENAI_API_KEY=$OPENAI_API_KEY
UMLS_API_KEY=$UMLS_API_KEY
WHO_CLIENT_ID=$WHO_CLIENT_ID
WHO_CLIENT_SECRET=$WHO_CLIENT_SECRET
LOINC_USERNAME=$LOINC_USERNAME
LOINC_PASSWORD=$LOINC_PASSWORD
PINECONE_API_KEY=$PINECONE_API_KEY
PINECONE_INDEX_NAME=$PINECONE_INDEX_NAME
PINECONE_ENVIRONMENT=$PINECONE_ENVIRONMENT
RXNORM_BASE_URL=https://rxnav.nlm.nih.gov/REST
ICD10CM_BASE_URL=https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search
EOF
)"

echo "✅ Deployment complete!"
echo "🌐 Your app should be available shortly"