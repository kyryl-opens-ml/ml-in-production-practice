import duckdb
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DuckDB")
conn = duckdb.connect(':memory:')

@mcp.tool()
def execute_sql(sql: str) -> str:
    """Executes a SQL query against the DuckDB database."""
    try:
        result = conn.execute(sql).fetchall()
        return json.dumps(result)
    except Exception as e:
        return f"Error executing SQL: {str(e)}"

@mcp.tool()
def get_tables() -> str:
    """Retrieves the list of tables in the DuckDB database."""
    try:
        tables = conn.execute("SHOW TABLES;").fetchall()
        return json.dumps(tables)
    except Exception as e:
        return f"Error retrieving tables: {str(e)}"
                
# Example: Create a sample table for testing
conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, name VARCHAR);")
conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');")

if __name__ == "__main__":
    mcp.run(transport="sse")
