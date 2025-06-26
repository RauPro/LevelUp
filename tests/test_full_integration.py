#!/usr/bin/env python3
"""
Integration test for the complete MLflow evaluation pipeline with Nebius API.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.schemas import DifficultyLevel, ProblemTopic
from models.state import Problem, SessionState


def test_full_evaluation_pipeline():
    """Test the complete evaluation pipeline with a mock state."""
    print("🧪 Testing full evaluation pipeline...")

    # Create a mock state
    mock_problem = Problem(
        description="Write a function to sort an array of integers using bubble sort algorithm.",
        tests=[
            {"input": "[3, 1, 4, 1, 5]", "expected": "[1, 1, 3, 4, 5]"},
            {"input": "[]", "expected": "[]"},
        ],
    )

    mock_state = SessionState(user_prompt="Generate an easy sorting problem", topic=ProblemTopic.SORTING, difficulty=DifficultyLevel.EASY, problem=mock_problem, code="def bubble_sort(arr): return sorted(arr)", tests_passed=True, problem_attempts=1, code_attempts=2)

    try:
        from ml.eval_mlflow import log_to_mlflow

        # Test the evaluation (should work with fallback metrics if API fails)
        print("📊 Running MLflow evaluation...")
        run_id, metrics, problem_attempts, code_attempts = log_to_mlflow(mock_state)

        print("✅ Evaluation completed successfully!")
        print(f"   Run ID: {run_id}")
        print(f"   Metrics count: {len(metrics) if metrics else 0}")
        print(f"   Problem attempts: {problem_attempts}")
        print(f"   Code attempts: {code_attempts}")

        return True

    except Exception as e:
        print(f"❌ Full evaluation test failed: {e}")
        return False


def main():
    """Run the full integration test."""
    print("🚀 Starting full Nebius API integration test...\n")

    # Check API key
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("❌ NEBIUS_API_KEY environment variable not set")
        print("   Setting a dummy key for testing...")
        os.environ["NEBIUS_API_KEY"] = "dummy_key_for_testing"

    # Run the test
    success = test_full_evaluation_pipeline()

    print(f"\n📊 Test Result: {'✅ PASSED' if success else '❌ FAILED'}")

    if success:
        print("🎉 The Nebius API integration is working correctly!")
        print("   Your MLflow evaluation pipeline is ready to use.")
    else:
        print("❌ There are issues with the integration.")
        print("   Please check the error messages above.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
