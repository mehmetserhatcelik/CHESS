import os
import sqlite3
import tempfile
from typing import Dict, Any, List

from llm.models import get_llm_chain, async_llm_chain_call
from llm.prompts import get_prompt
from llm.parsers import get_parser
from runner.database_manager import DatabaseManager
from workflow.system_state import SystemState
from workflow.agents.tool import Tool


class MockDatabaseGenerator(Tool):
    """
    Tool that builds a small mock SQLite DB from the full schema description and question intent.
    It creates tables, ensures join columns are consistent, and inserts up to 30 rows total with
    a few satisfying rows tracked in state.mock_satisfying_row_counts.
    """

    def __init__(self, template_name: str = "mock_db_generate", engine_config: Dict[str, Any] = None, parser_name: str = "list_output_parser"):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config or {"engine_name": "gemini-2.0-flash-lite", "temperature": 0.0}
        self.parser_name = parser_name

    def _run(self, state: SystemState) -> Dict:
        # Get complete schema string including types/descriptions
        database_schema = state.get_schema_string(schema_type="complete", include_value_description=True)

        request_kwargs = {
            "DATABASE_SCHEMA": database_schema,
            "QUESTION": state.task.question,
            "HINT": state.task.evidence,
            "MAX_ROWS": 30,
        }

        response = async_llm_chain_call(
            prompt=get_prompt(template_name=self.template_name),
            engine=get_llm_chain(**self.engine_config),
            parser=get_parser(self.parser_name),
            request_list=[request_kwargs],
            step=self.tool_name,
        )[0]

        # Response is expected to be a dict-like structure holding DDL, inserts, satisfying counts
        # Normalize expected keys
        ddl_statements: List[str] = response.get("ddl", [])
        insert_statements: List[str] = response.get("inserts", [])
        satisfying_counts: Dict[str, int] = response.get("satisfying_row_counts", {})
        generated_tables: Dict[str, List[Dict[str, Any]]] = response.get("generated_tables", {})

        # Create temp sqlite db
        tmp_dir = tempfile.mkdtemp(prefix="chess_mock_db_")
        mock_db_path = os.path.join(tmp_dir, "mock.sqlite")
        with sqlite3.connect(mock_db_path, timeout=30) as conn:
            cursor = conn.cursor()
            for ddl in ddl_statements:
                cursor.execute(ddl)
            for ins in insert_statements:
                cursor.execute(ins)
            conn.commit()

        state.mock_db_path = mock_db_path
        state.mock_satisfying_row_counts = satisfying_counts or {}
        state.mock_generated_tables = generated_tables or {}

    def _get_updates(self, state: SystemState) -> Dict:
        return {
            "node_type": self.tool_name,
            "mock_db_path": state.mock_db_path,
            "satisfying_row_counts": state.mock_satisfying_row_counts,
        }


