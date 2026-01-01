#!/bin/bash
# Deploy Face Mask Detection to Google Cloud Run
# Usage: ./deploy.sh <project-id> <backend-url>

set -e

PROJECT_ID=${1:-"invoismart"}
REGION="us-central1"

echo "🚀 Deploying Face Mask Detection to Google Cloud Run"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# ===== Deploy Backend =====
echo ""
echo "📦 Building and deploying backend..."
cd backend

# Build and push image
gcloud builds submit --tag gcr.io/$PROJECT_ID/mask-detection-api

# Deploy to Cloud Run
gcloud run deploy mask-detection-api \
    --image gcr.io/$PROJECT_ID/mask-detection-api \
    --platform managed \
    --region $REGION \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --allow-unauthenticated \
    --set-env-vars="PYTHONUNBUFFERED=1,ALLOW_ALL_ORIGINS=true"

# Get backend URL
BACKEND_URL=$(gcloud run services describe mask-detection-api --platform managed --region $REGION --format 'value(status.url)')
echo "✅ Backend deployed: $BACKEND_URL"

cd ..

# ===== Deploy Frontend =====
echo ""
echo "📦 Building and deploying frontend..."
cd frontend

# Create .env.production with backend URL
echo "VITE_API_URL=${BACKEND_URL}/api" > .env.production

# Build and push image
gcloud builds submit --tag gcr.io/$PROJECT_ID/mask-detection-frontend

# Deploy to Cloud Run
gcloud run deploy mask-detection-frontend \
    --image gcr.io/$PROJECT_ID/mask-detection-frontend \
    --platform managed \
    --region $REGION \
    --memory 256Mi \
    --allow-unauthenticated

# Get frontend URL
FRONTEND_URL=$(gcloud run services describe mask-detection-frontend --platform managed --region $REGION --format 'value(status.url)')
echo "✅ Frontend deployed: $FRONTEND_URL"

cd ..

# ===== Update Backend CORS =====
echo ""
echo "🔧 Updating backend CORS for frontend URL..."
# Note: You may need to manually update CORS_ORIGINS in config.py

echo ""
echo "=========================================="
echo "🎉 Deployment Complete!"
echo "=========================================="
echo ""
echo "Frontend: $FRONTEND_URL"
echo "Backend:  $BACKEND_URL"
echo ""
echo "⚠️  Remember to add $FRONTEND_URL to CORS_ORIGINS in backend/app/config.py if needed"
