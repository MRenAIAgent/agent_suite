#!/bin/bash

# Run Agent Comparison Script
# This script runs a comparison between custom React agent and LangChain's ReAct agent

# Set default values
MODEL="gpt-4"
CATEGORIES="reasoning planning tool_use"
TASK_LIMIT=3
OUTPUT_DIR="agent_comparison_results"
SKIP_LANGCHAIN=false
LANGCHAIN_ONLY=false

# Display header
echo "==============================================="
echo "  Agent Comparison with TauBench Evaluation"
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
    --skip-langchain)
      SKIP_LANGCHAIN=true
      shift
      ;;
    --langchain-only)
      LANGCHAIN_ONLY=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check for conflicting options
if [ "$SKIP_LANGCHAIN" = true ] && [ "$LANGCHAIN_ONLY" = true ]; then
  echo "Error: Cannot use both --skip-langchain and --langchain-only at the same time."
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting comparison with:"
echo "Model: $MODEL"
echo "Categories: $CATEGORIES"
echo "Task limit: $TASK_LIMIT"
echo "Results will be saved to: $OUTPUT_DIR"
if [ "$SKIP_LANGCHAIN" = true ]; then
  echo "Skipping LangChain agent evaluation"
fi
if [ "$LANGCHAIN_ONLY" = true ]; then
  echo "Evaluating only LangChain agent"
fi
echo ""

# Prepare command
CMD="python compare_agents.py --model \"$MODEL\" --categories $CATEGORIES --task-limit \"$TASK_LIMIT\" --output-dir \"$OUTPUT_DIR\""

# Add flags if needed
if [ "$SKIP_LANGCHAIN" = true ]; then
  CMD="$CMD --skip-langchain"
fi
if [ "$LANGCHAIN_ONLY" = true ]; then
  CMD="$CMD --langchain-only"
fi

# Run the comparison script
eval $CMD
RESULT=$?

# Check if comparison was successful
if [ $RESULT -eq 0 ]; then
  echo ""
  echo "Comparison completed successfully!"
  if [ "$SKIP_LANGCHAIN" = false ] && [ "$LANGCHAIN_ONLY" = false ]; then
    echo "Open $OUTPUT_DIR/comparison_report.html to view detailed results"
  elif [ "$LANGCHAIN_ONLY" = true ]; then
    echo "LangChain agent evaluation results are in $OUTPUT_DIR/langchain_react/"
  else
    echo "Custom agent evaluation results are in $OUTPUT_DIR/custom_react/"
  fi
else
  echo ""
  echo "Comparison encountered an error. Please check the logs."
  exit $RESULT
fi 