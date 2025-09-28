from tools import ToolsController, ToolResult
from agent import Agent
from pydantic import BaseModel
import asyncio

class CheckWeather(BaseModel):
    place: str

class ExecDb(BaseModel):
    query: str

async def main():
    controller = ToolsController()

    @controller.registry.tool("Check weather", param_model=CheckWeather)
    async def check_weather(params: CheckWeather) -> ToolResult:
        return ToolResult(content="Weather is hot, about 35 degree celcius")

    @controller.registry.tool("Interact with database", param_model=ExecDb)
    async def sql(params: ExecDb) -> ToolResult:
        return ToolResult(content="Successfully executed query")
    
    agent = Agent("Check weather of sansfransisco and insert to my db", controller=controller)
    asyncio.run(agent.run())

if __name__ == "__main__":
    asyncio.run(main())
else:
    raise RuntimeError("Examples cannot be imported")