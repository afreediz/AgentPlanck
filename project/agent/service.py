from tools import ToolsController, ToolResult
from langchain_openai import AzureChatOpenAI
from typing import Type, TypeVar
from pydantic import BaseModel
from tools.views import AgentOutput
from agent.message_manager import MessageManager

import os
import logging
logging.basicConfig(
    level=logging.DEBUG,  # or logging.INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)

llm = AzureChatOpenAI(
    model=os.environ.get('MODEL_NAME'), 
    api_key=os.environ.get('AZURE_OPENAI_KEY'),
    azure_endpoint=os.environ.get('AZURE_OPEAI_BASE'),
    api_version=os.environ.get('AZURE_OPENAI_VERSION')
)

class Agent():
    def __init__(self, task, tools_controller: ToolsController = ToolsController()):
        self.llm = llm
        self.tools_controller = tools_controller

        self.ToolOutputModel = self.tools_controller.registry.create_tool_model()
        self.AgentOutput = AgentOutput.type_with_custom_tools(self.ToolOutputModel)

        self.message_manager = MessageManager(
            task,
            tools_description=self.tools_controller.registry.get_prompt_description()
        )

    async def get_structured_response(self, messages, response_model:  Type[T]) -> T:
        """Get structured response using LangChain"""
        try:
            response = await self.llm.with_structured_output(response_model).ainvoke(messages)
            return response
        except Exception as e:
            raise e

    async def run(self) -> None:
        while True:
            messages = self.message_manager.get_messages()
            next_action = await self.get_structured_response(messages, response_model=self.AgentOutput)
            logger.info(f'\n\nPrev goal: {next_action.evaluation_previous_goal}\nMemory: {next_action.memory}\nNext goal: {next_action.next_goal}')
            logger.info(f'choosed : {next_action.choice.model_dump()}')

            self.message_manager.add_model_output(next_action)

            result = await self.tools_controller.act(next_action.choice)
            logger.info(f"result : {result.content}")

            if result.is_done:
                logger.info(f"\n\nCompleted!, {result.content}")
                logger.info(f"total tokens: {self.message_manager.history.total_tokens}")
                break
            
            self.message_manager.add_response(result.content)