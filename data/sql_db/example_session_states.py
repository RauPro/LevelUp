#!/usr/bin/env python3
"""
Example script demonstrating how to use the session state database functions.
This script shows how to save, retrieve, and query session states.
"""

import sys
import os
from uuid import uuid4

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data.sql_db.session_state_db import (
    save_session_state,
    get_session_state,
    get_session_states_by_criteria
)
from models.state import Problem
from app.api.schemas import ProblemTopic, DifficultyLevel


def create_example_session_state():
    """Create an example session state for testing"""
    return {
        "user_prompt": "Create a problem about finding the maximum element in an array",
        "topic": ProblemTopic.ARRAYS,
        "difficulty": DifficultyLevel.EASY,
        "problem": Problem(
            description="Find the maximum element in an array of integers.",
            tests=[
                {"input": "5 1 3 9 2", "output": "9"},
                {"input": "10 20 5", "output": "20"},
                {"input": "7", "output": "7"}
            ]
        ),
        "code": """
def find_max():
    numbers = list(map(int, input().split()))
    return max(numbers)

print(find_max())
""".strip(),
        "tests_passed": True,
        "code_attempts": 2,
        "problem_attempts": 1
    }


def demonstrate_save_and_retrieve():
    """Demonstrate saving and retrieving a session state"""
    print("🔄 Demonstrating save and retrieve functionality...")
    
    # Create a sample session state
    thread_id = str(uuid4())
    run_id = f"mlflow_run_{uuid4()}"
    session_state = create_example_session_state()
    
    print(f"📝 Saving session state with thread_id: {thread_id}")
    
    # Save the session state
    record_id = save_session_state(
        thread_id=thread_id,
        state=session_state,
        run_id=run_id,
        workflow_status='completed'
    )
    
    if record_id:
        print(f"✅ Session saved with record ID: {record_id}")
        
        # Retrieve the session state
        print(f"🔍 Retrieving session state...")
        retrieved_state = get_session_state(thread_id)
        
        if retrieved_state:
            print("✅ Session retrieved successfully!")
            print(f"   - Topic: {retrieved_state['topic']}")
            print(f"   - Difficulty: {retrieved_state['difficulty']}")
            print(f"   - Tests Passed: {retrieved_state['tests_passed']}")
            print(f"   - Code Attempts: {retrieved_state['code_attempts']}")
            print(f"   - Problem Attempts: {retrieved_state['problem_attempts']}")
            print(f"   - Workflow Status: {retrieved_state['workflow_status']}")
        else:
            print("❌ Failed to retrieve session state")
    else:
        print("❌ Failed to save session state")
    
    return thread_id


def demonstrate_queries():
    """Demonstrate querying session states with different criteria"""
    print("\n🔍 Demonstrating query functionality...")
    
    # Create a few more example sessions with different characteristics
    test_sessions = [
        {
            "thread_id": str(uuid4()),
            "state": {
                "user_prompt": "Create a string manipulation problem",
                "topic": ProblemTopic.STRINGS,
                "difficulty": DifficultyLevel.MEDIUM,
                "problem": Problem(
                    description="Reverse a string.",
                    tests=[{"input": "hello", "output": "olleh"}]
                ),
                "code": "print(input()[::-1])",
                "tests_passed": True,
                "code_attempts": 1,
                "problem_attempts": 1
            },
            "workflow_status": 'completed'
        },
        {
            "thread_id": str(uuid4()),
            "state": {
                "user_prompt": "Create a hard dynamic programming problem",
                "topic": ProblemTopic.DYNAMIC_PROGRAMMING,
                "difficulty": DifficultyLevel.HARD,
                "problem": None,  # No problem generated
                "code": None,
                "tests_passed": False,
                "code_attempts": 5,
                "problem_attempts": 2
            },
            "workflow_status": 'failed'
        }
    ]
    
    # Save the test sessions
    for session in test_sessions:
        save_session_state(
            thread_id=session["thread_id"],
            state=session["state"],
            workflow_status=session["workflow_status"]
        )
    
    # Query by topic
    print("📋 Querying sessions by topic 'arrays':")
    array_sessions = get_session_states_by_criteria(topic='arrays', limit=10)
    print(f"   Found {len(array_sessions)} array sessions")
    
    # Query by difficulty
    print("📋 Querying sessions by difficulty 'easy':")
    easy_sessions = get_session_states_by_criteria(difficulty='easy', limit=10)
    print(f"   Found {len(easy_sessions)} easy sessions")
    
    # Query by success status
    print("📋 Querying successful sessions:")
    successful_sessions = get_session_states_by_criteria(tests_passed=True, limit=10)
    print(f"   Found {len(successful_sessions)} successful sessions")
    
    # Query by workflow status
    print("📋 Querying completed sessions:")
    completed_sessions = get_session_states_by_criteria(workflow_status='completed', limit=10)
    print(f"   Found {len(completed_sessions)} completed sessions")



def main():
    """Main function to run all demonstrations"""
    print("🚀 Session State Database Demonstration")
    print("=" * 50)
    
    try:
        # Demonstrate basic save and retrieve
        thread_id = demonstrate_save_and_retrieve()
        
        # Demonstrate queries
        demonstrate_queries()
        
        
        print("\n🎉 All demonstrations completed successfully!")
        print(f"\n💡 Pro tip: You can view your data in the database using:")
        print("   SELECT * FROM session_states_summary ORDER BY created_at DESC;")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Make sure:")
        print("   1. Database is running and accessible")
        print("   2. session_states table has been initialized")
        print("   3. Environment variables are set correctly")


if __name__ == "__main__":
    main()
