# Nebius API Integration for MLflow Evaluation

This document describes the implementation of custom API key support for Nebius API in the LevelUp MLflow evaluation system.

## Overview

The implementation replaces the default OpenAI integration with a custom wrapper that supports the Nebius API while maintaining compatibility with MLflow's evaluation framework.

## Key Components

### 1. NebiusWrapper (`ml/nebius_wrapper.py`)

A custom MLflow PyFunc model that:
- Handles authentication with Nebius API using custom API key
- Implements the MLflow PythonModel interface
- Makes HTTP requests to Nebius API endpoints
- Provides error handling and fallback responses

### 2. Updated Metrics (`ml/metrics.py`)

Modified the existing metrics to:
- Use custom API configuration for Nebius
- Set proper OpenAI-compatible parameters for MLflow evaluation
- Include the custom API key and base URL

### 3. Updated Evaluation Logic (`ml/eval_mlflow.py`)

Enhanced the evaluation function to:
- Use the custom NebiusWrapper instead of direct OpenAI integration
- Set environment variables for OpenAI client compatibility
- Provide fallback metrics in case of evaluation failures
- Maintain compatibility with existing Grafana metrics

## Configuration

### Environment Variables

Make sure the following environment variable is set:

```bash
export NEBIUS_API_KEY="your_nebius_api_key_here"
```

### API Configuration

The implementation uses:
- **Base URL**: `https://api.studio.nebius.ai/v1`
- **Model**: `Qwen/QwQ-32B`
- **Temperature**: `0.0` for consistent evaluation

## Usage

The integration is transparent to existing code. Simply call `log_to_mlflow()` as before:

```python
from ml.eval_mlflow import log_to_mlflow

# Your existing code
run_id, metrics, problem_attempts, code_attempts = log_to_mlflow(state, state_history)
```

## Testing

Run the integration test to verify everything works:

```bash
python ml/test_nebius_integration.py
```

## Error Handling

The implementation includes multiple fallback mechanisms:

1. **API Request Failures**: Returns error messages instead of crashing
2. **Evaluation Failures**: Falls back to default metric values (3.0)
3. **Missing API Key**: Graceful degradation with informative error messages

## Metrics

The system evaluates two main metrics:

1. **Difficulty Accuracy**: How well the generated problem matches the requested difficulty
2. **Topic Relevance**: How well the problem aligns with the requested topic/algorithm

Both metrics are scored on a scale of 1-5 and include aggregations (mean, variance, p90).

## Dependencies

Added to `pyproject.toml`:
- `requests>=2.31.0` for HTTP API calls

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Verify your `NEBIUS_API_KEY` is set correctly
2. **Network Issues**: Check connectivity to `api.studio.nebius.ai`
3. **Model Errors**: Ensure the model `Qwen/QwQ-32B` is available in your Nebius account

### Debug Mode

Enable detailed logging by setting:

```bash
export MLFLOW_VERBOSE=true
```

## Future Improvements

- Add support for multiple models
- Implement retry logic for failed API calls
- Add rate limiting and quota management
- Support for streaming responses
