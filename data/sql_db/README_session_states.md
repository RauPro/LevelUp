# Session State Database Documentation

This module provides database functionality for storing and retrieving session states from the AI agent workflow.

## Overview

The session state system captures and stores complete information about each problem generation workflow, including:

- User input and preferences
- Generated problem details
- Code solutions
- Test results and attempts
- Workflow status and completion

## Database Schema

### Table: `session_states`

| Column                  | Type         | Description                           |
| ----------------------- | ------------ | ------------------------------------- |
| `id`                  | UUID         | Primary key (auto-generated)          |
| `created_at`          | TIMESTAMP    | Record creation time                  |
| `updated_at`          | TIMESTAMP    | Last update time (auto-updated)       |
| `thread_id`           | VARCHAR(255) | Unique workflow thread identifier     |
| `run_id`              | VARCHAR(255) | MLflow run ID for correlation         |
| `user_prompt`         | TEXT         | User's problem request                |
| `topic`               | VARCHAR(50)  | Problem topic (arrays, strings, etc.) |
| `difficulty`          | VARCHAR(20)  | Difficulty level (easy, medium, hard) |
| `problem_description` | TEXT         | Generated problem description         |
| `problem_tests`       | JSONB        | Array of test cases                   |
| `code`                | TEXT         | Generated solution code               |
| `tests_passed`        | BOOLEAN      | Whether tests passed                  |
| `code_attempts`       | INTEGER      | Number of code generation attempts    |
| `problem_attempts`    | INTEGER      | Number of problem generation attempts |
| `workflow_status`     | VARCHAR(20)  | running, completed, or failed         |

## Setup

### 1. Initialize the Database Table

```bash
cd data/sql_db
python init_session_states.py
```

This will create:

- The `session_states` table
- All necessary indexes
- Triggers for automatic timestamp updates
- The `session_states_summary` view

### 2. Environment Variables

Ensure these environment variables are set:

```bash
POSTGRES_HOST=localhost
POSTGRES_DB=neondb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_PORT=5432
```

## Usage

### Python API

```python
from data.sql_db.session_state_db import (
    save_session_state,
    get_session_state,
    get_session_states_by_criteria,
    get_session_analytics
)

# Save a session state
record_id = save_session_state(
    thread_id="unique-thread-id",
    state=session_state_dict,
    run_id="mlflow-run-id",
    workflow_status="completed"
)

# Retrieve a specific session
session_data = get_session_state("thread-id")

# Query sessions with criteria
sessions = get_session_states_by_criteria(
    topic="arrays",
    difficulty="easy",
    tests_passed=True,
    limit=50
)

# Get analytics
analytics = get_session_analytics()
```

### REST API Endpoints

The following endpoints are available in the FastAPI application:

#### Get Session by Thread ID

```
GET /api/sessions/{thread_id}
```

#### Query Sessions with Filters

```
GET /api/sessions?topic=arrays&difficulty=easy&tests_passed=true&limit=20
```

Query parameters:

- `topic`: Filter by problem topic
- `difficulty`: Filter by difficulty level
- `workflow_status`: Filter by workflow status
- `tests_passed`: Filter by test success status
- `limit`: Maximum number of results (default: 50, max: 100)

## Integration with Agent Workflow

The session state is automatically saved in the `/problems/verified` endpoint:

1. A unique `thread_id` is generated for each workflow
2. The agent runs with this thread ID
3. Upon completion, the final state is saved with correlation to MLflow run ID
4. The workflow status is set based on whether tests passed

## Database Functions

### Core Functions

- `save_session_state()`: Save or update a session state
- `get_session_state()`: Retrieve a session by thread ID
- `get_session_states_by_criteria()`: Query sessions with filters
- `delete_session_state()`: Delete a session by thread ID

### Utility Functions

- `get_db_connection()`: Get database connection using environment variables

## Example Usage

See `example_session_states.py` for a complete demonstration of all functionality:

```bash
cd data/sql_db
python example_session_states.py
```

## Monitoring and Analytics

### Key Metrics Tracked

1. **Success Rate**: Percentage of sessions with passing tests
2. **Average Attempts**: Mean code and problem generation attempts
3. **Topic Distribution**: Which topics are most requested
4. **Difficulty Distribution**: Which difficulty levels are most common
5. **Workflow Status**: Completion vs failure rates

### Debug Mode

Enable debug logging by setting environment variable:

```bash
export DEBUG_SESSION_STATES=1
```

## Performance Considerations

### Indexes

The table includes indexes on frequently queried columns:

- `thread_id` (unique)
- `run_id`
- `created_at`
- `topic`
- `difficulty`
- `workflow_status`

## Future Enhancements

Potential improvements:

1. **Session Clustering**: Group related sessions
2. **Performance Metrics**: Track execution times
3. **User Attribution**: Link sessions to specific users
4. **Version Tracking**: Track changes to problems and code
5. **Export Functionality**: Export session data for analysis
6. **Real-time Monitoring**: Live dashboards for session metrics
