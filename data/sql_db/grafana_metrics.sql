CREATE TABLE grafana_metrics (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    experiment_id VARCHAR(255),

    -- Topic relevance metrics
    topic_relevance_mean DECIMAL(10, 6),
    topic_relevance_variance DECIMAL(10, 6),
    topic_relevance_p90 DECIMAL(10, 6),

    -- Difficulty accuracy metrics
    difficulty_accuracy_mean DECIMAL(10, 6),
    difficulty_accuracy_variance DECIMAL(10, 6),
    difficulty_accuracy_p90 DECIMAL(10, 6),

    -- Additional metadata
    problem_type VARCHAR(100),
    evaluation_status VARCHAR(50) DEFAULT 'completed',
);
