#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Create required directories
mkdir -p uploads
mkdir -p uploads/extracted_files

# Create models directory and copy models
mkdir -p models
cp ../models/* models/
