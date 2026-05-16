#!/bin/bash

TIME_STAMP=$(date +%Y%m%d_%H%M%S)

# Azure OpenAI Configuration
AZURE_API_KEY="${AZURE_API_KEY:-YOUR_API_KEY}"
AZURE_ENDPOINT="${AZURE_ENDPOINT:-YOUR_API_ENDPOINT}"
AZURE_API_VERSION="2024-03-01-preview"
MODEL_NAME="${MODEL_NAME:-YOUR_MODEL_DEPLOYMENT_NAME}"  # Azure OpenAI deployment name (or export MODEL_NAME)

# Experiment Configuration
DATA_LIMIT=500
NUM_WORKERS=30
TEMPERATURE=0.2
MAX_LENGTH=32768
AUTO_TEST=1  # Set to 1 to enable automatic testing, 0 to disable

# Create output directory
output_dir=run_${TIME_STAMP}
mkdir -p ${output_dir}

echo "======================================"
echo "Azure OpenAI Experiment Setup"
echo "======================================"
echo "Timestamp: ${TIME_STAMP}"
echo "Model: ${MODEL_NAME}"
echo "Output Directory: ${output_dir}"
echo "Data Limit: ${DATA_LIMIT}"
echo "Number of Workers: ${NUM_WORKERS}"
echo "Temperature: ${TEMPERATURE}"
echo "Max Length: ${MAX_LENGTH}"
echo "Auto Test: ${AUTO_TEST}"
echo "======================================"

# Copy source files to output directory
echo "Copying source files to ${output_dir}/"
cp inference.py ${output_dir}/inference.py
cp utils.py ${output_dir}/utils.py
cp main.py ${output_dir}/main.py
cp test.py ${output_dir}/test.py 2>/dev/null || echo "test.py not found, skipping"
echo "Done copying files"

# Create model-specific output directory
model_output_dir=${output_dir}/${MODEL_NAME}_${TIME_STAMP}_${TEMPERATURE}
mkdir -p ${model_output_dir}

echo ""
echo "Starting Azure OpenAI experiment..."
echo "Model output directory: ${model_output_dir}"

# Run with Azure OpenAI API
nohup python main.py \
    --output_dir ${model_output_dir} \
    --data_path ../data.json \
    --data_limit ${DATA_LIMIT} \
    --model ${MODEL_NAME} \
    --use_azure 1 \
    --azure_api_key ${AZURE_API_KEY} \
    --azure_endpoint ${AZURE_ENDPOINT} \
    --azure_api_version ${AZURE_API_VERSION} \
    --temperature ${TEMPERATURE} \
    --max_length ${MAX_LENGTH} \
    --num_workers ${NUM_WORKERS} \
    --auto_test ${AUTO_TEST} \
    > ${model_output_dir}/log.log 2>&1 &

# Save the process ID
PROCESS_ID=$!
echo ${PROCESS_ID} > ${model_output_dir}/process.pid

echo ""
echo "======================================"
echo "Experiment Started Successfully!"
echo "======================================"
echo "Process ID: ${PROCESS_ID}"
echo "Log file: ${model_output_dir}/log.log"
echo ""
echo "To monitor progress, run:"
echo "  tail -f ${model_output_dir}/log.log"
echo ""
echo "To check process status, run:"
echo "  ps -p ${PROCESS_ID}"
echo ""
echo "To stop the experiment, run:"
echo "  kill ${PROCESS_ID}"
echo "======================================"