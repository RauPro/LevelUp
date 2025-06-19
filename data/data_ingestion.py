import json
import logging
import os
import subprocess

import numpy as np
import psycopg2
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from pandas.core.interchange.dataframe_protocol import DataFrame
from psycopg2.extras import execute_values

from data.preprocessing import process_codeforces_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Get database configuration from environment variables
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'dbname': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
}

def download_codeforces_dataset(cache_dir: str = "./hf_cache") -> DatasetDict:
    try:
        logging.info("Starting download of the dataset"
                     " 'evanellis/codeforces_with_only_correct_completions'...")

        # Ensure the cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Load the dataset using the Hugging Face library
        dataset = load_dataset(
            "evanellis/codeforces_with_only_correct_completions",
            cache_dir=cache_dir
        )

        logging.info("The dataset has been downloaded and loaded successfully.")
        logging.info(f"Dataset info: {dataset}")

        return dataset

    except Exception as e:
        logging.error(f"Error downloading the dataset from Hugging Face: {e}")
        raise

def save_to_postgres(df: DataFrame, table_name: str) -> any:
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            contestId TEXT,
            idx TEXT,
            name TEXT,
            rating INT,
            tags TEXT,
            problem_description TEXT,
            input_specification TEXT,
            output_specification TEXT,
            demo_input TEXT,
            demo_output TEXT,
            note TEXT,
            test_cases TEXT,
            code TEXT
        )
    """)
    records = df[['contestId', 'index', 'name', 'rating', 'tags',
                  'problem-description', 'input-specification',
                  'output-specification', 'demo-input',
                  'demo-output', 'note', 'code']].values.tolist()
    def adapt_record(record: list) -> any:
        def convert_value(v: any) -> any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, dict):
                return json.dumps(v)
            else:
                return v
        return [convert_value(v) for v in record]
    records = [adapt_record(row) for row in records]
    execute_values(cur,
        f"INSERT INTO {table_name} "
        f"(contestId, idx, name, rating, tags, problem_description, "
        f"input_specification, output_specification, demo_input, demo_output, "
        f"note, code) VALUES %s",
        records
    )
    conn.commit()
    cur.close()
    conn.close()

def export_table_as_sql(table_name: str, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "pg_dump",
        "-h", POSTGRES_CONFIG['host'],
        "-U", POSTGRES_CONFIG['user'],
        "-d", POSTGRES_CONFIG['dbname'],
        "-t", table_name,
        "-f", output_path
    ]
    env = os.environ.copy()
    env["PGPASSWORD"] = POSTGRES_CONFIG['password']
    subprocess.run(cmd, check=True, env=env)


if __name__ == '__main__':
    print("Running the data ingestion script as a standalone program...")
    raw_dataset = download_codeforces_dataset()
    print("\nExample of a record from the downloaded dataset:")
    if 'train' in raw_dataset:
        print(raw_dataset['train'][0])
        df = raw_dataset['train'].to_pandas()
        df = process_codeforces_data(df)
        save_to_postgres(df)
        print("\nTrain split saved to PostgreSQL table 'codeforces_train'")
        # Export the table as SQL for DVC tracking
        export_table_as_sql('codeforces_train', 'sql_db/codeforces_train.sql')
        print("Table exported as SQL to data/codeforces_train.sql")
    else:
        print("The 'train' split was not found in the dataset.")
