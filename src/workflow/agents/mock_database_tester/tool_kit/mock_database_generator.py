from typing import Dict, Any, List
import sqlite3
import tempfile

from pydantic import BaseModel

from llm.models import call_llm_chain, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.agents.tool import Tool


class MockDatabaseGenerator(Tool):
    """Generate a small mock database based on schema and question."""

    class Config(BaseModel):
        template_name: str
        engine_config: Dict[str, Any]
        parser_name: str

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.config = self.Config(**kwargs)
        self.generated_sql: List[str] = []
        self.satisfying_rows: Dict[str, Any] = {}

    def _run(self, state: SystemState):
        request_kwargs = {
            "DATABASE_SCHEMA": state.get_schema_string(schema_type="complete"),
            "QUESTION": state.task.question,
        }
        response = call_llm_chain(
            prompt=get_prompt(template_name=self.config.template_name),
            engine=get_llm_chain(**self.config.engine_config),
            parser=get_parser(self.config.parser_name),
            request_kwargs=request_kwargs,
            step=f"{self.tool_name}",
        )
        self.generated_sql = response.get("sql_statements", [])
        self.satisfying_rows = response.get("satisfying_rows", {})

        # Create temporary database and execute statements
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        db_path = tmp.name
        tmp.close()
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            for stmt in self.generated_sql:
                cursor.execute(stmt)
            conn.commit()
        finally:
            conn.close()
        state.mock_db_path = db_path
        state.mock_db_sqls = self.generated_sql
        state.satisfying_rows = self.satisfying_rows

    def _get_updates(self, state: SystemState) -> Dict[str, Any]:
        return {
            "node_type": self.tool_name,
            "mock_db_path": state.mock_db_path,
            "sql_count": len(self.generated_sql),
        }
