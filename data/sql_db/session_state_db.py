import json
import os
import psycopg2
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from models.state import SessionState, Problem
from app.api.schemas import ProblemTopic, DifficultyLevel


def get_db_connection():
    """Get database connection using environment variables"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        database=os.getenv('POSTGRES_DB', 'neondb'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'password'),
        port=os.getenv('POSTGRES_PORT', '5432')
    )


def save_session_state(
    thread_id: str,
    state: SessionState,
    run_id: Optional[str] = None,
    workflow_status: str = 'running'
) -> Optional[str]:
    """
    Save session state to database
    
    Args:
        thread_id: Unique identifier for the workflow thread
        state: The SessionState object containing all workflow data
        run_id: Optional MLflow run ID for correlation
        workflow_status: Status of the workflow (running, completed, failed)
    
    Returns:
        The UUID of the saved record, or None if save failed
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Convert problem tests to JSON format
        problem_tests_json = None
        if state.get("problem") and state["problem"].tests:
            problem_tests_json = json.dumps(state["problem"].tests)

        # Insert or update session state
        insert_query = """
        INSERT INTO session_states (
            thread_id, run_id, user_prompt, topic, difficulty,
            problem_description, problem_tests, code, tests_passed,
            code_attempts, problem_attempts, workflow_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (thread_id) DO UPDATE SET
            run_id = EXCLUDED.run_id,
            problem_description = EXCLUDED.problem_description,
            problem_tests = EXCLUDED.problem_tests,
            code = EXCLUDED.code,
            tests_passed = EXCLUDED.tests_passed,
            code_attempts = EXCLUDED.code_attempts,
            problem_attempts = EXCLUDED.problem_attempts,
            workflow_status = EXCLUDED.workflow_status,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
        """

        values = (
            thread_id,
            run_id,
            state["user_prompt"],
            state["topic"].value if isinstance(state["topic"], ProblemTopic) else state["topic"],
            state["difficulty"].value if isinstance(state["difficulty"], DifficultyLevel) else state["difficulty"],
            state["problem"].description if state.get("problem") else None,
            problem_tests_json,
            state.get("code"),
            state.get("tests_passed", False),
            state.get("code_attempts", 0),
            state.get("problem_attempts", 0),
            workflow_status
        )

        cur.execute(insert_query, values)
        record_id = cur.fetchone()[0]
        conn.commit()

        print(f"✅ Saved session state for thread_id: {thread_id}")
        return str(record_id)

    except Exception as e:
        print(f"❌ Error saving session state: {e}")
        return None
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()


def get_session_state(thread_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve session state from database by thread_id
    
    Args:
        thread_id: The thread ID to search for
        
    Returns:
        Dictionary containing session state data, or None if not found
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        query = """
        SELECT id, thread_id, run_id, created_at, updated_at,
               user_prompt, topic, difficulty, problem_description,
               problem_tests, code, tests_passed, code_attempts,
               problem_attempts, workflow_status
        FROM session_states
        WHERE thread_id = %s
        """

        cur.execute(query, (thread_id,))
        result = cur.fetchone()

        if result:
            columns = [
                'id', 'thread_id', 'run_id', 'created_at', 'updated_at',
                'user_prompt', 'topic', 'difficulty', 'problem_description',
                'problem_tests', 'code', 'tests_passed', 'code_attempts',
                'problem_attempts', 'workflow_status'
            ]
            
            session_data = dict(zip(columns, result))
            
            # Parse JSON fields - PostgreSQL JSONB returns Python objects directly
            # Only parse if it's actually a string (shouldn't happen with JSONB)
            if session_data['problem_tests'] and isinstance(session_data['problem_tests'], str):
                session_data['problem_tests'] = json.loads(session_data['problem_tests'])
            
            return session_data
        
        return None

    except Exception as e:
        print(f"❌ Error retrieving session state: {e}")
        return None
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()


def get_session_states_by_criteria(
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    workflow_status: Optional[str] = None,
    tests_passed: Optional[bool] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Retrieve session states based on criteria
    
    Args:
        topic: Filter by topic (optional)
        difficulty: Filter by difficulty (optional)
        workflow_status: Filter by workflow status (optional)
        tests_passed: Filter by tests_passed status (optional)
        limit: Maximum number of records to return
        
    Returns:
        List of dictionaries containing session state data
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Build dynamic query based on criteria
        where_conditions = []
        params = []

        if topic:
            where_conditions.append("topic = %s")
            params.append(topic)
        
        if difficulty:
            where_conditions.append("difficulty = %s")
            params.append(difficulty)
        
        if workflow_status:
            where_conditions.append("workflow_status = %s")
            params.append(workflow_status)
        
        if tests_passed is not None:
            where_conditions.append("tests_passed = %s")
            params.append(tests_passed)

        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        query = f"""
        SELECT id, thread_id, run_id, created_at, updated_at,
               user_prompt, topic, difficulty, problem_description,
               problem_tests, code, tests_passed, code_attempts,
               problem_attempts, workflow_status
        FROM session_states
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT %s
        """

        params.append(limit)
        cur.execute(query, params)
        results = cur.fetchall()

        columns = [
            'id', 'thread_id', 'run_id', 'created_at', 'updated_at',
            'user_prompt', 'topic', 'difficulty', 'problem_description',
            'problem_tests', 'code', 'tests_passed', 'code_attempts',
            'problem_attempts', 'workflow_status'
        ]

        session_list = []
        for result in results:
            session_data = dict(zip(columns, result))
            
            # Parse JSON fields - PostgreSQL JSONB returns Python objects directly
            # Only parse if it's actually a string (shouldn't happen with JSONB)
            if session_data['problem_tests'] and isinstance(session_data['problem_tests'], str):
                session_data['problem_tests'] = json.loads(session_data['problem_tests'])
            
            session_list.append(session_data)

        return session_list

    except Exception as e:
        print(f"❌ Error retrieving session states: {e}")
        return []
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()



def delete_session_state(thread_id: str) -> bool:
    """
    Delete a session state by thread_id
    
    Args:
        thread_id: The thread ID to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("DELETE FROM session_states WHERE thread_id = %s", (thread_id,))
        deleted_count = cur.rowcount
        conn.commit()

        if deleted_count > 0:
            print(f"✅ Deleted session state for thread_id: {thread_id}")
            return True
        else:
            print(f"⚠️ No session state found for thread_id: {thread_id}")
            return False

    except Exception as e:
        print(f"❌ Error deleting session state: {e}")
        return False
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()
