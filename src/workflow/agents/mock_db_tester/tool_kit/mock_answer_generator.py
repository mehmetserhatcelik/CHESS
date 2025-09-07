from typing import Dict, Any
import sqlite3

from llm.models import get_llm_chain, async_llm_chain_call
from llm.prompts import get_prompt
from llm.parsers import get_parser
from runner.database_manager import DatabaseManager
from workflow.system_state import SystemState
from workflow.agents.tool import Tool


class MockAnswerGenerator(Tool):
    """
    Tool that infers the expected answer table (shape and rows) from the question and mock DB contents,
    then persists it both as JSON in state.mock_expected_answer and as a physical table named `answer` in the mock DB.
    """

    def __init__(self, template_name: str = "mock_answer_generate", engine_config: Dict[str, Any] = None, parser_name: str = "select_tables"):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config or {"engine_name": "gemini-2.0-flash-lite", "temperature": 0.0}
        self.parser_name = parser_name

    def _run(self, state: SystemState) -> Dict:
        # Gather minimal info for LLM to infer answer shape and rows
        schema_for_candidates = state.get_database_schema_for_queries(
            queries=[sql_meta.SQL for sql_meta in state.SQL_meta_infos.get("generate_candidate", [])] or [state.task.SQL]
        )

        request_kwargs = {
            "DATABASE_SCHEMA": schema_for_candidates,
            "QUESTION": state.task.question,
            "HINT": state.task.evidence,
            "SATISFYING_ROW_COUNTS": state.mock_satisfying_row_counts,
        }

        response = async_llm_chain_call(
            prompt=get_prompt(template_name=self.template_name),
            engine=get_llm_chain(**self.engine_config),
            parser=get_parser("select_tables"),
            request_list=[request_kwargs],
            step=self.tool_name,
        )[0]

        # Expected response example:
        # {"attributes": ["col1", "col2"], "values": [[v11, v12], [v21, v22]]}
        answer_json = {"attributes": response.get("attributes", []), "values": response.get("values", [])}
        state.mock_expected_answer = answer_json

        # Create answer table in mock DB
        if state.mock_db_path and answer_json.get("attributes"):
            columns = ", ".join([f"`{c}` TEXT" for c in answer_json["attributes"]])
            create_sql = f"CREATE TABLE IF NOT EXISTS `answer` ({columns});"
            with sqlite3.connect(state.mock_db_path, timeout=30) as conn:
                cur = conn.cursor()
                cur.execute(create_sql)
                for row in answer_json.get("values", []):
                    literal_values = ", ".join([f"'{str(v)}'" for v in row])
                    literal_insert = f"INSERT INTO `answer` VALUES ({literal_values});"
                    cur.execute(literal_insert)
                conn.commit()

    def _get_updates(self, state: SystemState) -> Dict:
        return {
            "node_type": self.tool_name,
            "answer": state.mock_expected_answer,
        }


