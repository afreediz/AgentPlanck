from planck import Agent, ToolsController, ToolResult
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import asyncio
import os
load_dotenv()

llm = AzureChatOpenAI(
    model=os.environ.get('MODEL_NAME'), 
    api_key=os.environ.get('AZURE_OPENAI_KEY'),
    azure_endpoint=os.environ.get('AZURE_OPENAI_BASE'),
    api_version=os.environ.get('AZURE_OPENAI_VERSION')
)
# llm = ChatOpenAI(model="gpt-4o")

class CheckWeather(BaseModel):
    city: str

class ExecDb(BaseModel):
    query: str

async def main():
    tools_controller = ToolsController()

    @tools_controller.registry.tool("Check weather", param_model=CheckWeather)
    async def check_weather(params: CheckWeather) -> ToolResult:
        return ToolResult(content="Weather is hot, about 35°C")

    @tools_controller.registry.tool("Interact with database", param_model=ExecDb)
    async def sql(params: ExecDb) -> ToolResult:
        return ToolResult(content="Successfully executed query")
    
    agent = Agent(
        "Check weather of San Francisco and insert to my db", 
        llm=llm, 
        tools_controller=tools_controller
    )
    res = await agent.run()
    print(res)

if __name__ == "__main__":
    asyncio.run(main())