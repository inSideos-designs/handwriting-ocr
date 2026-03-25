#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="doc-processing-hw-ocr"
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER_NAME="handwriting-ocr-cluster"
BUCKET_NAME="doc-processing-hw-ocr-ml"
REPO_NAME="handwriting-ocr"

echo "Step 1: Create GCP project"
gcloud projects create "$PROJECT_ID" --name="Document Processing" || echo "Project may already exist"

echo "Step 2: Set project"
gcloud config set project "$PROJECT_ID"

echo "Step 3: Link billing"
echo "If billing is not linked, run:"
echo "  gcloud billing accounts list"
echo "  gcloud billing projects link $PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID"
read -p "Press enter once billing is linked..."

echo "Step 4: Enable APIs"
gcloud services enable \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    compute.googleapis.com \
    cloudbuild.googleapis.com \
    --project="$PROJECT_ID"

echo "Step 5: Create GCS bucket"
gcloud storage buckets create "gs://$BUCKET_NAME" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --uniform-bucket-level-access \
    || echo "Bucket may already exist"

echo "Step 6: Create Artifact Registry repository"
gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION" \
    --project="$PROJECT_ID" \
    || echo "Repository may already exist"

echo "Step 7: Configure Docker auth for Artifact Registry"
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

echo "Step 8: Create GKE cluster with CPU node pool"
gcloud container clusters create "$CLUSTER_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --num-nodes=1 \
    --machine-type=e2-small \
    --disk-size=30 \
    --enable-autoscaling --min-nodes=1 --max-nodes=2 \
    --workload-pool="$PROJECT_ID.svc.id.goog" \
    --scopes=storage-full

echo "Step 9: Add GPU node pool"
gcloud container node-pools create gpu-pool \
    --cluster="$CLUSTER_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --num-nodes=0 \
    --enable-autoscaling --min-nodes=0 --max-nodes=1 \
    --spot \
    --scopes=storage-full

echo "Step 10: Install NVIDIA GPU device plugin"
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

echo "Step 11: Get cluster credentials"
gcloud container clusters get-credentials "$CLUSTER_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID"

echo "Step 12: Create dagster namespace"
kubectl create namespace dagster || echo "Namespace may already exist"

echo ""
echo "Infrastructure setup complete!"
echo ""
echo "Next steps:"
echo "  1. Upload Kaggle dataset to GCS:"
echo "     gsutil -m rsync -r /Users/cultistsid/Downloads/archive/ gs://$BUCKET_NAME/data/raw/"
echo "  2. Build and push Docker image:"
echo "     gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/dagster:latest"
echo "  3. Install Dagster via Helm:"
echo "     helm repo add dagster https://dagster-io.github.io/helm"
echo "     helm install dagster dagster/dagster -n dagster -f infra/dagster-values.yaml"
