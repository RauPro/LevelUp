import psycopg2
import os
import numpy as np
from datetime import datetime

from ml.agent import EXPERIMENT_NAME

def get_db_connection():
    """Get database connection using environment variables"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        database=os.getenv('POSTGRES_DB', 'neondb'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'password'),
        port=os.getenv('POSTGRES_PORT', '5432')
    )

def save_evaluation_results(run_id, metrics_dict, problem_attempts, code_attempts, experiment_id=EXPERIMENT_NAME):
    """
    Save complete evaluation results to database
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Convert numpy types to Python types
        metrics = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (np.float64, np.float32)):
                metrics[key] = float(value)
            elif isinstance(value, (np.int64, np.int32)):
                metrics[key] = int(value)
            else:
                metrics[key] = value

        insert_query = """
        INSERT INTO evaluation_metrics (
            run_id, experiment_id,
            topic_relevance_mean, topic_relevance_variance, topic_relevance_p90,
            difficulty_accuracy_mean, difficulty_accuracy_variance, difficulty_accuracy_p90,
            problem_attempts, code_attempts
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            topic_relevance_mean = EXCLUDED.topic_relevance_mean,
            topic_relevance_variance = EXCLUDED.topic_relevance_variance,
            topic_relevance_p90 = EXCLUDED.topic_relevance_p90,
            difficulty_accuracy_mean = EXCLUDED.difficulty_accuracy_mean,
            difficulty_accuracy_variance = EXCLUDED.difficulty_accuracy_variance,
            difficulty_accuracy_p90 = EXCLUDED.difficulty_accuracy_p90,
            problem_attempts = EXCLUDED.problem_attempts,
            code_attempts = EXCLUDED.code_attempts,
            timestamp = CURRENT_TIMESTAMP
        """

        values = (
            run_id,
            experiment_id,
            metrics.get('topic_relevance/v1/mean'),
            metrics.get('topic_relevance/v1/variance'),
            metrics.get('topic_relevance/v1/p90'),
            metrics.get('difficulty_accuracy/v1/mean'),
            metrics.get('difficulty_accuracy/v1/variance'),
            metrics.get('difficulty_accuracy/v1/p90'),
            problem_attempts,
            code_attempts
        )

        cur.execute(insert_query, values)
        conn.commit()

        print(f"✅ Saved evaluation results for run_id: {run_id}")
        return True

    except Exception as e:
        print(f"❌ Error saving evaluation results: {e}")
        return False
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()

