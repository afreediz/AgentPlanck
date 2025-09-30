from typing import Tuple, Any
from uuid import uuid4

def get_key_param(model: dict) -> Tuple[Any, Any]:
    for key, value in model.items():
        key_name = key
        param = value
        break

    return key_name, param

def generate_random(max_length:int = 5) -> str:
    return uuid4().hex[:max_length+1]