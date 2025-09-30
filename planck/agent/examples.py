from tools import ToolsController, ToolResult
from agent import Agent
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import asyncio
import os

load_dotenv()

llm = AzureChatOpenAI(
    model=os.environ.get('MODEL_NAME'), 
    api_key=os.environ.get('AZURE_OPENAI_KEY'),
    azure_endpoint=os.environ.get('AZURE_OPENAI_BASE'),
    api_version=os.environ.get('AZURE_OPENAI_VERSION')
)
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