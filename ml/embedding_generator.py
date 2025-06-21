import os
from pathlib import Path
from typing import Any, Optional

import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from tqdm import tqdm  # type: ignore


class EmbeddingGenerator:
    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize the embedding generator with proper persistence setup

        Args:
            db_path: Path to ChromaDB storage. If None, uses absolute path in current directory
        """
        # Use absolute path to ensure consistency
        if db_path is None:
            self.db_path = os.path.abspath("../data/chroma_db")
        else:
            self.db_path = os.path.abspath(db_path)

        # Ensure the directory exists
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        print(f"Using ChromaDB path: {self.db_path}")

        # Initialize ChromaDB client with absolute path
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Create an embedding function using a pre-trained model
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        # Create or get the collection
        self.problem_collection = self.client.get_or_create_collection(
            name="problems",
            embedding_function=self.sentence_transformer_ef,
        )

        print(f"Collection initialized with {self.problem_collection.count()} existing documents")

    def add_to_chroma_in_batches(self, df: Any, batch_size: int = 5000) -> int:  # noqa: ANN401
        """
        Add dataframe records to ChromaDB collection in batches to avoid exceeding max batch size

        Args:
            df: pandas DataFrame with the data to add
            batch_size: maximum number of records to add in a single batch
        """
        initial_count = self.problem_collection.count()
        total_batches = (len(df) + batch_size - 1) // batch_size

        for i in tqdm(range(total_batches), desc="Adding to vector DB"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            # Prepare the batch data for ChromaDB
            documents = batch_df["problem-description"].tolist()

            metadatas = batch_df[["contestId", "index", "name", "rating", "tags"]].to_dict("records")
            # Convert any NumPy arrays to strings in the metadata
            for metadata in metadatas:
                if isinstance(metadata["tags"], np.ndarray):
                    metadata["tags"] = ", ".join(metadata["tags"].tolist())
                elif isinstance(metadata["tags"], list):
                    metadata["tags"] = ", ".join(metadata["tags"])

            # Create unique IDs - using batch index to avoid conflicts
            ids = [f"{row['contestId']}-{row['index']}-{start_idx + j}" for j, (_, row) in enumerate(batch_df.iterrows())]

            try:
                # Add the batch to the collection
                self.problem_collection.add(ids=ids, documents=documents, metadatas=metadatas)
                print(f"Batch {i + 1}/{total_batches} added successfully")
            except Exception as e:
                print(f"Error adding batch {i + 1}: {e}")
                continue

        final_count = self.problem_collection.count()
        added_count = final_count - initial_count
        print(f"Successfully added {added_count} new records to ChromaDB in {total_batches} batches")
        print(f"Total documents in collection: {final_count}")

        return final_count

    def get_collection_info(self) -> int:
        """Get information about the current collection"""
        count = self.problem_collection.count()
        print(f"Collection 'problems' has {count} documents")
        print(f"Database path: {self.db_path}")

        # List all collections to verify
        collections = self.client.list_collections()
        print(f"Available collections: {[col.name for col in collections]}")

        return count

    def clear_collection(self) -> None:
        """Clear all data from the collection (use with caution!)"""
        try:
            self.client.delete_collection("problems")
            print("Collection 'problems' deleted")

            # Recreate the collection
            self.problem_collection = self.client.get_or_create_collection(
                name="problems",
                embedding_function=self.sentence_transformer_ef,
            )
            print("Collection 'problems' recreated")
        except Exception as e:
            print(f"Error clearing collection: {e}")


# For backward compatibility, create a global instance
embedding_gen = EmbeddingGenerator()
client = embedding_gen.client
sentence_transformer_ef = embedding_gen.sentence_transformer_ef
problem_collection = embedding_gen.problem_collection


def add_to_chroma_in_batches(df: Any, batch_size: int = 5000) -> int:  # noqa: ANN401
    """Backward compatible function"""
    return embedding_gen.add_to_chroma_in_batches(df, batch_size)


if __name__ == "__main__":
    # Test the persistence
    print("=== Testing ChromaDB Persistence ===")

    # Create a new instance to test persistence
    test_gen = EmbeddingGenerator()
    count = test_gen.get_collection_info()

    # If you want to test with sample data, uncomment below:
    # sample_df = pd.DataFrame({
    #     'problem-description': ['Sample problem 1', 'Sample problem 2'],
    #     'contestId': [1, 2],
    #     'index': ['A', 'B'],
    #     'name': ['Problem A', 'Problem B'],
    #     'rating': [1000, 1200],
    #     'tags': [['math'], ['implementation']]
    # })
    # test_gen.add_to_chroma_in_batches(sample_df)
