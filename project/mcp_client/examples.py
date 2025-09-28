from tools import ToolsController
from mcp_client.service import MCPClient
from agent import Agent
import asyncio


async def main() -> None:
    # Connect to an MCP server
    controller = ToolsController()
    mcp_client = MCPClient(
        server_name="my-server",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem",r"C:\Users\Afree\Desktop\New folder"]
    )

    # Register all MCP tools as actions
    await mcp_client.register_to_controller(controller)
    agent = Agent("Create a file named sample.txt and write 100 words poem to it", tools_controller=controller)
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())