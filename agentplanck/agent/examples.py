from agentplanck.tools import ToolsController, ToolResult
from agentplanck.agent import Agent
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

class CheckWeather(BaseModel):
    place: str

class ExecDb(BaseModel):
    query: str

async def main():
    controller = ToolsController(handle_tools_error=False)

    @controller.registry.tool("Check weather", param_model=CheckWeather)
    async def check_weather(params: CheckWeather) -> ToolResult:
        return ToolResult(content="Weather is hot, about 35 degree celcius")

    @controller.registry.tool("Interact with database", param_model=ExecDb)
    async def sql(params: ExecDb) -> ToolResult:
        return ToolResult(content="Successfully executed query")
    
    agent = Agent("Check weather of sansfransisco and insert to my db", llm=llm, tools_controller=controller)
    res = await agent.run()
    print(res)

if __name__ == "__main__":
    asyncio.run(main())
else:
    raise RuntimeError("Examples cannot be imported")