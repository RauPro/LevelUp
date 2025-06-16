

def process_codeforces_data(df):
    print("\n--- 2. Filtering for Programming Problems ---")
    df = df[df['type'] == 'PROGRAMMING'].copy()
    print(f"Shape after filtering for 'PROGRAMMING' type: {df.shape}")

    print("\n--- 3. Dropping Unnecessary Columns ---")
    columns_to_drop = [
        'points', 'creationTimeSeconds', 'relativeTimeSeconds',
        'programmingLanguage', 'verdict', 'testset', 'passedTestCount',
        'timeConsumedMillis', 'memoryConsumedBytes', 'prompt', 'response',
        'score', 'state', 'correct_completion', 'id', 'type', "time-limit", "memory-limit", "title"
    ]

    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_columns_to_drop, inplace=True)
    print(f"Columns dropped. Final shape: {df.shape}")
    print(f"Remaining columns: {df.columns.tolist()}")

    print("\n--- 4. Data Summary and Null Value Analysis ---")

    print("Dataset Information:")
    df.info()


    print("\nNull Value Counts:")
    null_counts = df.isnull().sum()
    print(null_counts)

    print("\n--- 5. Dropping Rows with Null Values ---")
    critical_columns = ['problem-description', 'name', 'tags', 'rating', 'code']
    initial_rows = df.shape[0]
    df.dropna(subset=critical_columns, inplace=True)

    if initial_rows > df.shape[0]:
        print(f"Dropped {initial_rows - df.shape[0]} rows containing null values in critical columns.")
    else:
        print("No rows with null values in critical columns found.")

    print(f"Final cleaned shape: {df.shape}")

    print("\n--- 6. Sample of Final Cleaned Data ---")

    print(df.head())

    cleaned_csv_path = "codeforces_cleaned.csv"
    df.to_csv(cleaned_csv_path, index=False)
    return df

if __name__ == '__main__':
    print("Running the data preprocessing script as a standalone program...")
    #process_codeforces_data()
    print("Data preprocessing completed successfully.")
