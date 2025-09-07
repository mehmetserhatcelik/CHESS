from typing import Dict, Any, List, Tuple
import sqlite3

from workflow.system_state import SystemState
from workflow.agents.tool import Tool
from workflow.sql_meta_info import SQLMetaInfo


class Decision(Tool):
    """Select the candidate SQL matching the mock answer."""

    def __init__(self):
        super().__init__()
        self.selected_sql: str | None = None

    def _run(self, state: SystemState):
        expected = [tuple(row) for row in state.mock_answer.get("values", [])]
        expected_set = set(expected)
        candidates: List[str] = []
        for metas in state.SQL_meta_infos.values():
            for meta in metas:
                candidates.append(meta.SQL)
        conn = sqlite3.connect(state.mock_db_path)
        cursor = conn.cursor()
        for sql in candidates:
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                if set(result) == expected_set:
                    self.selected_sql = sql
                    break
            except Exception:
                continue
        conn.close()
        if self.selected_sql:
            state.SQL_meta_infos[self.tool_name] = [SQLMetaInfo(SQL=self.selected_sql)]
        else:
            state.errors[self.tool_name] = "No candidate SQL matched the expected answer"

    def _get_updates(self, state: SystemState) -> Dict[str, Any]:
        return {
            "node_type": self.tool_name,
            "selected_sql": self.selected_sql,
        }
