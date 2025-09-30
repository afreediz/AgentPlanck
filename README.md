# AgentPlanck

**AgentPlanck** is a lightweight agentic framework I built to understand and experiment with how agents work under the hood.

---

## ✨ Features
- **Dynamic Pydantic model generation** for structured outputs.  
- **Message managers** for clean communication flow.  
- **MCP handlers** for flexible tool interaction.  
- **Fault tolerance** mechanisms to keep agents running reliably.  
- And more to explore! 🚀  

---

## 🔬 Why "AgentPlanck"?  
The name comes from **Max Planck** and the **Planck constant**, which defines the smallest "action" in physics.  
It’s fitting for a system that takes **intelligent actions at minimal scales**.

---

## ⚡ Quick Start

```python
import asyncio
from tools import ToolsController, ToolResult
from agent import Agent
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

class CheckWeather(BaseModel):
    city: str

class ExecDb(BaseModel):
    query: str

async def main():
    tools_controller = ToolsController()

    @controller.registry.tool("Check weather", param_model=CheckWeather)
    async def check_weather(params: CheckWeather) -> ToolResult:
        return ToolResult(content="Weather is hot, about 35°C")

    @controller.registry.tool("Interact with database", param_model=ExecDb)
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
```

---


## 📜 License
MIT License. Free to use and modify.