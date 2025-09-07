from typing import Dict, Any, List
import sqlite3
from pydantic import BaseModel

from llm.models import call_llm_chain, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.agents.tool import Tool


class MockAnswerGenerator(Tool):
    """Generate the expected answer table for the mock database."""

    class Config(BaseModel):
        template_name: str
        engine_config: Dict[str, Any]
        parser_name: str

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.config = self.Config(**kwargs)
        self.answer: Dict[str, Any] = {}

    def _run(self, state: SystemState):
        request_kwargs = {
            "QUESTION": state.task.question,
            "SATISFYING_ROWS": state.satisfying_rows,
        }
        response = call_llm_chain(
            prompt=get_prompt(template_name=self.config.template_name),
            engine=get_llm_chain(**self.config.engine_config),
            parser=get_parser(self.config.parser_name),
            request_kwargs=request_kwargs,
            step=f"{self.tool_name}",
        )
        self.answer = response

        conn = sqlite3.connect(state.mock_db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS answer")
        cols = ", ".join([f"{c} TEXT" for c in self.answer.get("attributes", [])])
        cursor.execute(f"CREATE TABLE answer ({cols})")
        for row in self.answer.get("values", []):
            placeholders = ",".join(["?" for _ in row])
            cursor.execute(f"INSERT INTO answer VALUES ({placeholders})", row)
        conn.commit()
        conn.close()
        state.mock_answer = self.answer

    def _get_updates(self, state: SystemState) -> Dict[str, Any]:
        return {
            "node_type": self.tool_name,
            "answer": self.answer,
        }
