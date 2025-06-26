#!/usr/bin/env python3
"""
Script to initialize the session_states table in the database.
Run this script to create the table and related database objects.
"""

import psycopg2
import os
from pathlib import Path


def get_db_connection():
    """Get database connection using environment variables"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        database=os.getenv('POSTGRES_DB', 'neondb'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'password'),
        port=os.getenv('POSTGRES_PORT', '5432')
    )


def initialize_session_states_table():
    """Initialize the session_states table and related database objects"""
    try:
        # Read the SQL file
        sql_file_path = Path(__file__).parent / "session_states.sql"
        
        if not sql_file_path.exists():
            print(f"❌ SQL file not found: {sql_file_path}")
            return False
            
        with open(sql_file_path, 'r') as f:
            sql_content = f.read()

        # Connect to database and execute SQL
        conn = get_db_connection()
        cur = conn.cursor()
        
        print("🔄 Initializing session_states table...")
        
        # Execute the SQL commands
        cur.execute(sql_content)
        conn.commit()
        
        print("✅ Successfully initialized session_states table and related objects!")
        print("📊 Created:")
        print("   - session_states table")
        print("   - Indexes for better query performance")
        print("   - Trigger for automatic updated_at timestamp")
        print("   - session_states_summary view")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing session_states table: {e}")
        return False
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()


def check_table_exists():
    """Check if the session_states table already exists"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'session_states'
        );
        """)
        
        exists = cur.fetchone()[0]
        return exists
        
    except Exception as e:
        print(f"❌ Error checking table existence: {e}")
        return False
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()


def main():
    """Main function to run the initialization"""
    print("🚀 Session States Table Initialization")
    print("=" * 40)
    
    # Check if table already exists
    if check_table_exists():
        print("⚠️  session_states table already exists!")
        response = input("Do you want to recreate it? (This will drop existing data) [y/N]: ")
        
        if response.lower() != 'y':
            print("🛑 Initialization cancelled.")
            return
        else:
            # Drop existing table
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                print("🗑️  Dropping existing table...")
                cur.execute("DROP TABLE IF EXISTS session_states CASCADE;")
                conn.commit()
                print("✅ Existing table dropped.")
            except Exception as e:
                print(f"❌ Error dropping table: {e}")
                return
            finally:
                if 'conn' in locals():
                    cur.close()
                    conn.close()
    
    # Initialize the table
    if initialize_session_states_table():
        print("\n🎉 Session states table is ready for use!")
        print("\n📝 You can now:")
        print("   - Save session states using save_session_state()")
        print("   - Query sessions using get_session_state() or get_session_states_by_criteria()")
        print("   - Get analytics using get_session_analytics()")
        print("   - Use the new API endpoints: /api/sessions, /api/sessions/{thread_id}, /api/sessions/analytics")
    else:
        print("\n❌ Failed to initialize session states table.")


if __name__ == "__main__":
    main()
