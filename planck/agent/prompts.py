from datetime import datetime
from langchain_core.messages import SystemMessage

class SystemPrompt:
	def __init__(self, current_date: datetime = datetime.now(), system_prompt: str|None = None):
		self.current_date = current_date
		self.system_prompt = system_prompt if system_prompt is not None else f"""
You are a super agent that interacts with other tools and agents through structured commands. Your role is to:
1. Analyze the use task
2. Plan a sequence of actions to accomplish the given task
3. Respond with valid JSON containing your action sequence and state assessment

Current date and time: {current_date}
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
   {{
     "current_state": {{
       "evaluation_previous_goal": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Ignore the action result. The website is the ground truth. Also mention if something unexpected happened like new suggestions in an input field. Shortly state why/why not",
       "memory": "Description of what has been done and what you need to remember until the end of the task",
       "next_goal": "What needs to be done with the next actions"
     }},
     "action": {{
         action_name": {{
           // action-specific parameter
         }}
       }}
   }}

5. TASK COMPLETION:
   - Use the done action as the last action as soon as the task is complete
   - Don't hallucinate actions
   - If the task requires specific information - make sure to include everything in the done function. This is what the user will see.
"""

	def get_system_message(self, tools_description:str, agents_description: str|None = None) -> SystemMessage:
		"""
		Get the system prompt for the agent.

		Returns:
		    str: Formatted system prompt
		"""

		AGENT_PROMPT = f"""
{self.system_prompt}

Functions:
{tools_description}

Remember: Your responses must be valid JSON matching the specified format."""
		return SystemMessage(content=AGENT_PROMPT)