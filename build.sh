#!/bin/bash
# CycloNex — Render.com build script
# Builds React frontend then copies it into backend/static/frontend/
# Run from repo root directory

set -e

echo "=== Step 1: Install Python dependencies ==="
cd backend
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Step 2: Build React frontend ==="
cd ../frontend
npm ci
npm run build

echo "=== Step 3: Copy frontend build → backend/static/frontend/ ==="
mkdir -p ../backend/static/frontend
cp -r dist/* ../backend/static/frontend/

echo "=== Build complete ==="
echo "Frontend files:"
ls ../backend/static/frontend/
