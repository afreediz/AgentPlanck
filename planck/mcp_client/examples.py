from planck import ToolsController, MCPClient, Agent
from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

async def main() -> None:
    # Connect to an MCP server
    controller = ToolsController()
    mcp_client = MCPClient(
        server_name="my-server",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem",r"C:\\Users\\Afree\\Desktop\\test_mcp"]
    )

    # Register all MCP tools as actions
    await mcp_client.register_to_controller(controller)
    agent = Agent("Create a file named sample.txt and write 100 words poem to it", llm=llm, tools_controller=controller)
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())