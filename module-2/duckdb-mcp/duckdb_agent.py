import asyncio

from agents import Agent, ModelSettings, Runner
from agents.mcp import MCPServerSse


async def main():
    conversation_history = []

    async with MCPServerSse(
        name="DuckDB MCP Server",
        params={
            "url": "http://0.0.0.0:8000/sse",
        },
    ) as server:
        tools = await server.list_tools()
        print(f"Available tools: {[x.name for x in tools]}")

        print("\nExample queries you can ask:")
        print("- Show me the tables")
        print("- What is the schema for the users table?")
        print("- Show me all users")
        print("- Select the names of users with id greater than 1")

        agent = Agent(
            name="DuckDB Assistant",
            instructions="""You are a helpful assistant for interacting with a DuckDB database via MCP tools.
You can execute SQL queries and retrieve database schema information.
1. Use the 'get_tables' tool to see available tables.
2. Use the 'execute_sql' tool to run SQL queries (e.g., SELECT, SHOW, DESCRIBE).
3. Present query results clearly.
4. Ask for clarification if a query is ambiguous.""",
            mcp_servers=[server],
            model_settings=ModelSettings(tool_choice="auto", parallel_tool_calls=True),
        )

        while True:
            user_input = input("\nEnter your question (or 'exit' to quit): ")
            if user_input.lower() == "exit":
                break

            if conversation_history:
                context = "\n".join(conversation_history)
                full_input = (
                    f"Previous conversation:\n{context}\n\nNew question: {user_input}"
                )
            else:
                full_input = user_input

            print("\nProcessing your request...")
            result = await Runner.run(starting_agent=agent, input=full_input)
            print(f"\nResponse: {result.final_output}")

            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())