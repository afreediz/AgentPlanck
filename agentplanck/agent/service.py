from pydantic import BaseModel
from agentplanck.tools import ToolsController
from agentplanck.agent.views import AgentOutput, AgentResult
from agentplanck.agent.message_manager import MessageManager
from agentplanck.utils.general import generate_random
from typing import Type, TypeVar
from langchain.chat_models.base import BaseChatModel
import logging

logging.basicConfig(
    level=logging.INFO,  # or logging.INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)

class Agent():
    def __init__(self, task: str, llm: BaseChatModel, tools_controller: ToolsController = ToolsController()):
        self.llm:BaseChatModel = llm
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

    async def run(self) -> str:
        try:
            tool_errors_count = 0
            iteration = 1

            while True:
                logger.info(f"ITERATION : {iteration}")
                messages = self.message_manager.get_messages()
                next_action = await self.get_structured_response(messages, response_model=self.AgentOutput)

                tool_call_id = generate_random()
                self.message_manager.add_model_output(next_action, tool_call_id)

                logger.info(self.message_manager.format_agentoutput(next_action))

                result = await self.tools_controller.act(next_action.choice)
                logger.info(f"result : {result.content}")

                self.message_manager.add_response(result, tool_call_id)

                # Stop if 3 concecutive tool errors came to avoid looping
                if result.error is not None:
                    tool_errors_count += 1

                    if tool_errors_count > 2:
                        raise Exception(f"3 concecutive tool errors, exiting. Error: {result.error}")
                else:
                    tool_errors_count = 0

                iteration += 1
                if iteration % 10 == 0:
                    self.message_manager.cut_history()

                if result.is_done:
                    logger.info(f"\n\nCompleted!, {result.content}")
                    logger.info(f"total tokens: {self.message_manager.history.total_tokens}")
                    break

            return AgentResult(content=result.content, history=self.message_manager.get_all_messages(), tokens=self.message_manager.history.total_tokens)
        
        except Exception as e:
            current_messages = self.message_manager.get_messages(include_system_message=False)

            output_messages = []
            output_messages.append(current_messages[0])
            output_messages.extend(self.message_manager.dropped_message)
            output_messages.extend(current_messages[1:])

            return AgentResult(content=None, errors=str(e), history=self.message_manager.get_all_messages(), tokens=self.message_manager.history.total_tokens)