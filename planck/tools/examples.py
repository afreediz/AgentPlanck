from tools import ToolsController
from pydantic import BaseModel

class SimpleModel(BaseModel):
    query: str

def main():
    controller = ToolsController()

    @controller.registry.tool("Test func1", param_model=SimpleModel)
    async def func1(params: SimpleModel):
        print('execute func1')

    @controller.registry.tool("Test func2", param_model=SimpleModel)
    async def func2(params: SimpleModel):
        print("executed func2", params)

    res = controller.registry.get_prompt_description()
    print('desc ',res)

    res = controller.registry.create_tool_model()
    print('param model ', res.model_json_schema())

if __name__ == "__main__":
    main()
else:
    raise ValueError("examples cannot be imported")