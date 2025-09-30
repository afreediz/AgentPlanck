import json
import logging
from typing import List, Optional
from langchain_core.messages import (
	AIMessage,
	BaseMessage,
	HumanMessage,
	SystemMessage,
	ToolMessage,
)
from planck.utils.general import get_key_param
from planck.tools.views import ToolResult
from planck.agent.prompts import SystemPrompt
from planck.agent.views import AgentOutput
from planck.agent.message_manager.views import MessageHistory, MessageMetadata

logger = logging.getLogger(__name__)


class MessageManager:
	def __init__(
		self,
		task: str,
		max_input_tokens: int = 128000,
		max_error_length: int = 400,
		estimated_characters_per_token: int = 3,
		tools_description: str|None = None,
		system_prompt: SystemPrompt|None = None,
		message_context: Optional[str] = None,
	):
		self.max_input_tokens = max_input_tokens
		self.history = MessageHistory()
		self.task = task
		self.estimated_characters_per_token = estimated_characters_per_token
		self.max_error_length = max_error_length
		self.message_context = message_context

		self.IMG_TOKENS = 800 # rough estimate of tokens per image

		system_prompt = system_prompt if system_prompt is not None else SystemPrompt()
		system_message = system_prompt.get_system_message(tools_description=tools_description)

		self.dropped_message = []

		self.system_prompt = system_message
		self._add_message_with_tokens(system_message)

		if self.message_context:
			context_message = HumanMessage(content=self.message_context)
			self._add_message_with_tokens(context_message)

		task_message = self.task_instructions(task)
		self.add_human_message(task_message)
		self.tool_id = 1

	@staticmethod
	def task_instructions(task: str) -> str:
		content = f'Your ultimate task is: {task}. If you achieved your ultimate task, stop everything and use the done action in the next step to complete the task. If not, continue as usual.'
		return content
	
	def add_message(self, message: BaseMessage) -> None:
		self._add_message_with_tokens(message)

	def add_new_task(self, new_task: str) -> None:
		msg = HumanMessage(f'Your new ultimate task is: {new_task}. Take the previous context into account and finish your new ultimate task. ')
		self._add_message_with_tokens(msg)

	def add_human_message(self, message:str) -> None:
		self._add_message_with_tokens(
			HumanMessage(content=message)
		)

	def add_model_output(self, model_output: AgentOutput, tool_call_id: str) -> None:
		"""Add model output as AI message"""

		choice = model_output.choice.model_dump()
		key_name, param = get_key_param(choice)

		tool_calls = [
			{
				'name': key_name,
				'args': param,
				'id': tool_call_id,
				'type': 'tool_call',
			}
		]

		msg = AIMessage(
			content=self.format_agentoutput(model_output),
			tool_calls=tool_calls
		)
		self._add_message_with_tokens(msg)

	def add_response(self, message:ToolResult, call_id: int) -> None:
		if message.error:
			self._add_message_with_tokens(
				ToolMessage(
					content=message.error,
					tool_call_id=call_id
				)
			)
		else:
			self._add_message_with_tokens(
				ToolMessage(
					content=message.content,
					tool_call_id=call_id
				)
			)

	def format_agentoutput(self, model_output: AgentOutput) -> str:
		message = ""
		message += f"Prev goal: {model_output.evaluation_previous_goal},Memory: {model_output.memory},Next goal: {model_output.next_goal}. For this i will call:\n"
		message += model_output.choice.model_dump_json()
		return message

	def pretty_print_messages(self) -> None:
		logger.info('------------------------------------------------------------------------------------------------')
		for message in self.history.messages[2:]:
			message = message.message
			if isinstance(message, ToolMessage):
				logger.info(f"TOOL({message.name}): {message.content}")
			elif isinstance(message, AIMessage):
				logger.info(f"AI: {message.content}, calling tools: {', '.join([m.get('name') for m in message.tool_calls])}")
			elif isinstance(message, HumanMessage):
				logger.info(f"HUMAN: {message.content}")
			else:
				logger.info(f"{message.content}")
		logger.info('------------------------------------------------------------------------------------------------')

	def get_messages(self, include_system_message:bool = True) -> List[BaseMessage]:
		"""Get current message list, potentially trimmed to max tokens"""
		msg = [m.message for m in self.history.messages]
		# debug which messages are in history with token count # log
		total_input_tokens = 0
		logger.info(f'Messages in history: {len(self.history.messages)}:')
		for m in self.history.messages:
			total_input_tokens += m.metadata.input_tokens
			# logger.debug(f'{m.message.__class__.__name__} - Token count: {m.metadata.input_tokens}')

		logger.info(f'Total input tokens: {total_input_tokens}')

		return msg if include_system_message else msg[1:]

	def _add_message_with_tokens(self, message: BaseMessage) -> None:
		"""Add message with token count metadata"""
		token_count = self._count_tokens(message)
		metadata = MessageMetadata(input_tokens=token_count)
		self.history.add_message(message, metadata)

	def get_all_messages(self, include_system_message:bool = False) -> List[BaseMessage]:
		current_messages = self.get_messages(include_system_message=include_system_message)

		output_messages = []
		if include_system_message:
			output_messages.append(current_messages[0])
			output_messages.append(current_messages[1])
			output_messages.extend(self.dropped_message)
			output_messages.extend(current_messages[2:])
		else:
			output_messages.append(current_messages[0])
			output_messages.extend(self.dropped_message)
			output_messages.extend(current_messages[1:])

		return output_messages
	
	# later include more strategies
	def cut_history(self, max_messages:int = 20) -> None:
		if len(self.history.messages) <= 2:
			return  # nothing to cut

		# Always keep the first two - system message & human message discribing task
		fixed = self.history.messages[:2]
		rest = self.history.messages[2:]

		to_drop = rest[:-max_messages] if len(rest) > max_messages else []
		to_keep = rest[-max_messages:]

		# Backup the dropped part
		if to_drop:
			self.dropped_message.extend(to_drop)

		# Rebuild history
		self.history.messages = fixed + to_keep

	def _count_tokens(self, message: BaseMessage) -> int:
		"""Count tokens in a message using the model's tokenizer"""
		tokens = 0
		if isinstance(message.content, list):
			for item in message.content:
				if 'image_url' in item:
					tokens += self.IMG_TOKENS
				elif isinstance(item, dict) and 'text' in item:
					tokens += self._count_text_tokens(item['text'])
		else:
			msg = message.content
			if hasattr(message, 'tool_calls'):
				msg += str(message.tool_calls)  # type: ignore
			tokens += self._count_text_tokens(msg)
		return tokens

	def _count_text_tokens(self, text: str) -> int:
		"""Count tokens in a text string"""
		tokens = len(text) // self.estimated_characters_per_token  # Rough estimate if no tokenizer available
		return tokens

	def cut_messages(self):
		"""Get current message list, potentially trimmed to max tokens"""
		diff = self.history.total_tokens - self.max_input_tokens
		if diff <= 0:
			return None

		msg = self.history.messages[-1]

		# if list with image remove image
		if isinstance(msg.message.content, list):
			text = ''
			for item in msg.message.content:
				if 'image_url' in item:
					msg.message.content.remove(item)
					diff -= self.IMG_TOKENS
					msg.metadata.input_tokens -= self.IMG_TOKENS
					self.history.total_tokens -= self.IMG_TOKENS
					logger.debug(
						f'Removed image with {self.IMG_TOKENS} tokens - total tokens now: {self.history.total_tokens}/{self.max_input_tokens}'
					)
				elif 'text' in item and isinstance(item, dict):
					text += item['text']
			msg.message.content = text
			self.history.messages[-1] = msg

		if diff <= 0:
			return None

		# if still over, remove text from state message proportionally to the number of tokens needed with buffer
		# Calculate the proportion of content to remove
		proportion_to_remove = diff / msg.metadata.input_tokens
		if proportion_to_remove > 0.99:
			raise ValueError(
				f'Max token limit reached - history is too long - reduce the system prompt or task. '
				f'proportion_to_remove: {proportion_to_remove}'
			)
		logger.debug(
			f'Removing {proportion_to_remove * 100:.2f}% of the last message  {proportion_to_remove * msg.metadata.input_tokens:.2f} / {msg.metadata.input_tokens:.2f} tokens)'
		)

		content = msg.message.content
		characters_to_remove = int(len(content) * proportion_to_remove)
		content = content[:-characters_to_remove]

		# remove tokens and old long message
		self.history.remove_message(index=-1)

		# new message with updated content
		msg = HumanMessage(content=content)
		self._add_message_with_tokens(msg)

		last_msg = self.history.messages[-1]

		logger.debug(
			f'Added message with {last_msg.metadata.input_tokens} tokens - total tokens now: {self.history.total_tokens}/{self.max_input_tokens} - total messages: {len(self.history.messages)}'
		)

	def convert_messages_for_non_function_calling_models(self, input_messages: list[BaseMessage]) -> list[BaseMessage]:
		"""Convert messages for non-function-calling models"""
		output_messages = []
		for message in input_messages:
			if isinstance(message, HumanMessage):
				output_messages.append(message)
			elif isinstance(message, SystemMessage):
				output_messages.append(message)
			elif isinstance(message, ToolMessage):
				output_messages.append(HumanMessage(content=message.content))
			elif isinstance(message, AIMessage):
				# check if tool_calls is a valid JSON object
				if message.tool_calls:
					tool_calls = json.dumps(message.tool_calls)
					output_messages.append(AIMessage(content=tool_calls))
				else:
					output_messages.append(message)
			else:
				raise ValueError(f'Unknown message type: {type(message)}')
		return output_messages

	def merge_successive_human_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
		"""Some models like deepseek-reasoner dont allow multiple human messages in a row. This function merges them into one."""
		merged_messages:list[BaseMessage] = []
		streak = 0
		for message in messages:
			if isinstance(message, HumanMessage):
				streak += 1
				if streak > 1:
					merged_messages[-1].content += message.content
				else:
					merged_messages.append(message)
			else:
				merged_messages.append(message)
				streak = 0
		return merged_messages

	def extract_json_from_model_output(self, content: str) -> dict:
		"""Extract JSON from model output, handling both plain JSON and code-block-wrapped JSON."""
		try:
			# If content is wrapped in code blocks, extract just the JSON part
			if content.startswith('```'):
				# Find the JSON content between code blocks
				content = content.split('```')[1]
				# Remove language identifier if present (e.g., 'json\n')
				if '\n' in content:
					content = content.split('\n', 1)[1]
			# Parse the cleaned content
			return json.loads(content)
		except json.JSONDecodeError as e:
			logger.warning(f'Failed to parse model output: {str(e)}')
			raise ValueError('Could not parse response.')