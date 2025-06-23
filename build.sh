#!/bin/bash

# Railway Build Script for Medical Coding Intelligence Platform
# This script builds the React frontend and prepares it for FastAPI to serve

set -e  # Exit on any error

echo "🏗️  Building Medical Coding Intelligence Platform..."

# Step 1: Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install

# Step 2: Build React app
echo "⚛️  Building React application..."
npm run build

# Step 3: Prepare static files for FastAPI
echo "📁 Preparing static files for backend..."
cd ..
mkdir -p backend/static
cp -r frontend/build/* backend/static/

# Step 4: List what was built
echo "✅ Build completed!"
echo "📂 Static files copied to backend/static/"
ls -la backend/static/

echo "🚀 Ready for deployment!"