#!/bin/bash

# Usage: ./run_test_custom.sh <GPU_ID> <VAL_DATASET_DESCRIPTION> [MODEL_TYPE] [BASE_PATH] [ROUND]
# Example: ./run_test_custom.sh 0 "AC" "qwen3_vl"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <GPU_ID> [VAL_DATASET_DESCRIPTION] [MODEL_TYPE] [BASE_PATH] [ROUND]"
    echo "Example: $0 0 \"AC\" \"qwen3_vl\""
    echo "Note: VAL_DATASET_DESCRIPTION defaults to \"AC\""
    exit 1
fi

GPU_ID=$1
VAL_DATASET_DESCRIPTION=${2:-"AC"}
MODEL_TYPE=${3:-"qwen3_vl"}
# Generic paths substituted for privacy
BASE_PATH=${4:-"./output/fedavg/all_skew"}
ROUND=${5:-10}

# Construct validation dataset path
VAL_DATASET="./datasets/testset/${VAL_DATASET_DESCRIPTION}.json"

# Mapping PEFT ID to model descriptions
declare -A peft_to_description=(
    ["v109-20251227-203239"]="central cross-iid"
)

# List of PEFT IDs to process
peft_list=(
    v109-20251227-203239
)

for i in ${peft_list[@]};
do
    echo "Processing PEFT: $i"
    
    MODEL_DESCRIPTION=${peft_to_description[$i]}
    if [ -z "$MODEL_DESCRIPTION" ]; then
        echo "Warning: No description found for PEFT ID $i, skipping..."
        continue
    fi
    
    echo "Model Description: $MODEL_DESCRIPTION"
    
    # Generate output JSON filename: description_dataset.jsonl
    OUTPUT_JSON_NAME="${MODEL_DESCRIPTION}_${VAL_DATASET_DESCRIPTION}.jsonl"
    echo "Output Filename: $OUTPUT_JSON_NAME"
    
    # Construct model paths
    MODEL_PATH="$BASE_PATH/$MODEL_TYPE/$i/global_lora_$ROUND"
    INFER_RESULT_DIR="$MODEL_PATH/infer_result"
    
    echo "Model Path: $MODEL_PATH"
    echo "Inference Result Directory: $INFER_RESULT_DIR"
    
    # Check if inference results already exist for this validation set
    target_result_file="$INFER_RESULT_DIR/$OUTPUT_JSON_NAME"
    if [ -f "$target_result_file" ]; then
        echo "Found existing inference results: $target_result_file. Skipping inference..."
    else
        echo "Inference results not found. Starting inference..."
        if [ ! -f "$VAL_DATASET" ]; then
            echo "Error: Validation dataset file not found: $VAL_DATASET"
            continue
        fi
        
        # Execute inference
        MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=$GPU_ID swift infer --ckpt_dir "$MODEL_PATH" \
          --val_dataset "$VAL_DATASET" --model_type "$MODEL_TYPE" --sft_type lora
    fi
    
    # Process inference results
    if [ -f "$target_result_file" ]; then
        echo "Using existing result file: $target_result_file"
        result_file_to_process="$target_result_file"
    else
        # Find newly generated result file, excluding already named custom files
        jsonl_files=$(find "$INFER_RESULT_DIR" -type f -name "*.jsonl" ! -name "*_*.jsonl" 2>/dev/null)
        
        if [ -z "$jsonl_files" ]; then
            echo "Warning: No inference result files found in $INFER_RESULT_DIR"
            continue
        fi
        
        # Select the most recent jsonl file
        first_jsonl=$(find "$INFER_RESULT_DIR" -type f -name "*.jsonl" ! -name "*_*.jsonl" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
        
        if [ -z "$first_jsonl" ]; then
            echo "Warning: No suitable inference result file found."
            continue
        fi
        
        echo "Processing result file: $first_jsonl"
        
        # Wait for file write completion
        prev_size=0
        current_size=$(stat -c%s "$first_jsonl" 2>/dev/null || echo 0)
        while [ "$current_size" != "$prev_size" ]; do
            echo "Waiting for file write to complete..."
            sleep 2
            prev_size=$current_size
            current_size=$(stat -c%s "$first_jsonl" 2>/dev/null || echo 0)
        done
        
        # Create a copy with the custom description
        output_dir=$(dirname "$first_jsonl")
        result_file_to_process="$output_dir/$OUTPUT_JSON_NAME"
        
        # Safely copy file using a temporary file
        temp_file="${result_file_to_process}.tmp.$$"
        if cp "$first_jsonl" "$temp_file" && mv "$temp_file" "$result_file_to_process"; then
            echo "Created custom output file: $result_file_to_process"
        else
            echo "Error: Failed to copy file."
            rm -f "$temp_file"
            continue
        fi
    fi
    
    # Calculate accuracy
    echo "Calculating accuracy..."
    python ./evaluation/test.py --data_path "$result_file_to_process" --output_suffix "$VAL_DATASET_DESCRIPTION"
    
    echo "Finished processing: $result_file_to_process"
    echo "Results written to results_${VAL_DATASET_DESCRIPTION}.csv"
    echo "------------------------"
    
    echo "PEFT $i processing complete."
    echo "========================"
done

echo "All tasks finished!"
echo "Check results_${VAL_DATASET_DESCRIPTION}.csv for detailed metrics."