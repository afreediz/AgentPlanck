from pydantic import BaseModel
from enum import Enum
from pydantic import Field, BaseModel, create_model, ConfigDict
from typing import Union
from tools.views import ToolModel

class AgentOutput(BaseModel):
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

	evaluation_previous_goal: str
	memory: str
	next_goal: str
	choice: ToolModel = Field(
		...,
		description='tool to execute',
	)

	@staticmethod
	def type_with_custom_tools(tools_model: type[ToolModel]) -> type['AgentOutput']:
		"""Extend actions with custom actions"""

		model_ = create_model(
			'AgentOutput',
			__base__=AgentOutput,
			choice=(
				tools_model,  # type: ignore
				Field(..., description='tool to execute'),
			),
			__module__=AgentOutput.__module__,
		)
		model_.__doc__ = 'AgentOutput model with custom actions'
		return model_


class Result(BaseModel):
    content: str
    successfull: bool = False

class AgentResult(BaseModel):
    content: str
    successfull: str