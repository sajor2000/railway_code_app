#!/bin/bash
# Production deployment script for GCP Cloud Run

echo "🚀 Deploying Medical Coding Intelligence Platform - Production"

# Configuration
PROJECT_ID="medical-coding-intelligent"
SERVICE_NAME="medicalcodeassistant"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME-production"

# Build and push production image
echo "🔨 Building production Docker image..."
docker build -f Dockerfile.production -t $IMAGE_NAME .

echo "📤 Pushing image to Google Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run with production settings
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --set-env-vars "^|^$(cat <<EOF
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
EOF
)"

echo "✅ Production deployment complete!"
echo "🌐 Your app will be available at the URL provided by Cloud Run"