#!/bin/bash

# Run LangChain ReAct Agent Evaluation Script
# This script evaluates a LangChain ReAct agent using TauBench

# Set default values
MODEL="gpt-4.1"
CATEGORIES="reasoning planning tool_use"
TASK_LIMIT=3
OUTPUT_DIR="langchain_eval_results"
VERBOSE=false

# Display header
echo "==============================================="
echo "  LangChain ReAct Agent Evaluation with TauBench"
echo "==============================================="
echo ""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --categories)
      CATEGORIES="$2"
      shift 2
      ;;
    --task-limit)
      TASK_LIMIT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting LangChain agent evaluation with:"
echo "Model: $MODEL"
echo "Categories: $CATEGORIES"
echo "Task limit: $TASK_LIMIT"
echo "Results will be saved to: $OUTPUT_DIR"
if [ "$VERBOSE" = true ]; then
  echo "Verbose mode enabled"
fi
echo ""

# Prepare command
CMD="python langchain_react_eval.py --model \"$MODEL\" --categories $CATEGORIES --task-limit \"$TASK_LIMIT\" --output-dir \"$OUTPUT_DIR\""

# Add verbose flag if needed
if [ "$VERBOSE" = true ]; then
  CMD="$CMD --verbose"
fi

# Run the evaluation script
eval $CMD
RESULT=$?

# Check if evaluation was successful
if [ $RESULT -eq 0 ]; then
  echo ""
  echo "LangChain agent evaluation completed successfully!"
  echo "Results saved to: $OUTPUT_DIR"
else
  echo ""
  echo "Evaluation encountered an error. Please check the logs."
  exit $RESULT
fi 