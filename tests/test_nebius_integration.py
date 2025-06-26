#!/usr/bin/env python3
"""
Test script for the updated MLflow evaluation with Nebius API integration.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ml.metrics import difficulty_accuracy_metric, topic_relevance_metric
from ml.nebius_wrapper import NebiusWrapper


def test_nebius_wrapper():
    """Test the Nebius wrapper functionality."""
    print("🧪 Testing Nebius wrapper...")

    # Check if API key is set
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("❌ NEBIUS_API_KEY environment variable not set")
        return False

    print(f"✅ API key found: {api_key[:10]}...")

    # Test the wrapper
    wrapper = NebiusWrapper()
    test_input = pd.DataFrame({"inputs": ["Generate a simple sorting problem"]})

    try:
        result = wrapper.predict(None, test_input)
        print(f"✅ Wrapper test successful. Response length: {len(result[0]) if result else 0}")
        if result:
            print(f"📝 Sample response: {result[0][:100]}...")
        return True
    except Exception as e:
        print(f"❌ Wrapper test failed: {e}")
        return False


def test_metrics_configuration():
    """Test that metrics are properly configured."""
    print("\n🧪 Testing metrics configuration...")

    try:
        # Check if metrics have the right configuration
        assert difficulty_accuracy_metric.name == "difficulty_accuracy"
        assert topic_relevance_metric.name == "topic_relevance"

        # Check if metrics are callable evaluation metrics
        assert hasattr(difficulty_accuracy_metric, "name")
        assert hasattr(topic_relevance_metric, "name")

        print("✅ Metrics configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Metrics configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Nebius API integration tests...\n")

    tests = [
        test_metrics_configuration,
        test_nebius_wrapper,
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\n📊 Test Summary: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("🎉 All tests passed! The integration is ready to use.")
    else:
        print("❌ Some tests failed. Please check the configuration.")

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
