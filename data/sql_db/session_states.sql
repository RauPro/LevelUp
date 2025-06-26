-- Table for storing session states from the AI agent workflow
CREATE TABLE session_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Session identification
    thread_id VARCHAR(255) UNIQUE NOT NULL, -- From the agent workflow
    run_id VARCHAR(255), -- MLflow run ID for correlation with metrics
    
    -- User input
    user_prompt TEXT NOT NULL,
    topic VARCHAR(50) NOT NULL, -- arrays, strings, linked_lists, etc.
    difficulty VARCHAR(20) NOT NULL, -- easy, medium, hard
    
    -- Generated problem data
    problem_description TEXT,
    problem_tests JSONB, -- Array of test cases [{"input": "...", "output": "..."}]
    
    -- Generated code
    code TEXT,
    
    -- Workflow state
    tests_passed BOOLEAN DEFAULT FALSE,
    code_attempts INTEGER DEFAULT 0,
    problem_attempts INTEGER DEFAULT 0,
    
    -- Workflow completion status
    workflow_status VARCHAR(20) DEFAULT 'running', -- running, completed, failed
    
    -- Indexes for better query performance
    CONSTRAINT valid_topic CHECK (topic IN (
        'arrays', 'strings', 'linked_lists', 'trees', 'graphs', 
        'dynamic_programming', 'sorting', 'searching', 'recursion', 'backtracking'
    )),
    CONSTRAINT valid_difficulty CHECK (difficulty IN ('easy', 'medium', 'hard')),
    CONSTRAINT valid_workflow_status CHECK (workflow_status IN ('running', 'completed', 'failed'))
);

-- Create indexes for better query performance
CREATE INDEX idx_session_states_thread_id ON session_states(thread_id);
CREATE INDEX idx_session_states_run_id ON session_states(run_id);
CREATE INDEX idx_session_states_created_at ON session_states(created_at);
CREATE INDEX idx_session_states_topic ON session_states(topic);
CREATE INDEX idx_session_states_difficulty ON session_states(difficulty);
CREATE INDEX idx_session_states_workflow_status ON session_states(workflow_status);

-- Create a trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_session_states_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_session_states_updated_at
    BEFORE UPDATE ON session_states
    FOR EACH ROW
    EXECUTE FUNCTION update_session_states_updated_at();

-- Optional: Create a view for easier querying with formatted data
CREATE VIEW session_states_summary AS
SELECT 
    id,
    thread_id,
    run_id,
    created_at,
    updated_at,
    user_prompt,
    topic,
    difficulty,
    CASE 
        WHEN problem_description IS NOT NULL THEN 'Generated'
        ELSE 'Not Generated'
    END as problem_status,
    CASE 
        WHEN code IS NOT NULL THEN 'Generated'
        ELSE 'Not Generated'
    END as code_status,
    tests_passed,
    code_attempts,
    problem_attempts,
    workflow_status,
    jsonb_array_length(COALESCE(problem_tests, '[]'::jsonb)) as test_cases_count
FROM session_states
ORDER BY created_at DESC;
