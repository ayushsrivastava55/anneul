# Copyright Sierra (MIT). Trimmed to the two records the vendored task list needs.

from typing import Any

from pydantic import BaseModel


class Action(BaseModel):
    name: str
    kwargs: dict[str, Any]


class Task(BaseModel):
    user_id: str
    actions: list[Action]
    instruction: str
    outputs: list[str]
    annotator: str = ""
