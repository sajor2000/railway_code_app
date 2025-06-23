#!/bin/bash
# Test production server locally before deploying

echo "🧪 Testing Production V2 Server Locally"
echo "====================================="

# Build the image
echo "🔨 Building Docker image..."
docker build -f Dockerfile.production-v2 -t medical-prod-test .

# Run container
echo "🚀 Starting container..."
docker run -d --name medical-test -p 8080:8080 --env-file .env medical-prod-test

# Wait for startup
echo "⏳ Waiting for server to start..."
sleep 5

# Test health endpoint
echo "🏥 Testing health endpoint..."
curl -f http://localhost:8080/api/health | jq .

# Test search endpoint
echo -e "\n🔍 Testing search endpoint..."
curl -X POST http://localhost:8080/api/search/unified \
  -H "Content-Type: application/json" \
  -d '{"query": "aspirin", "ontologies": ["rxnorm"]}' | jq .

# Show logs
echo -e "\n📋 Container logs:"
docker logs medical-test | tail -20

# Cleanup
echo -e "\n🧹 Cleaning up..."
docker stop medical-test
docker rm medical-test

echo -e "\n✅ Test complete!"