from typing import Dict, Any, List, Tuple
import sqlite3

from runner.database_manager import DatabaseManager
from workflow.system_state import SystemState
from workflow.agents.tool import Tool


class MockSQLDecision(Tool):
    """
    Executes each candidate SQL on the mock database and compares results to the expected answer table.
    Picks the SQL whose result matches the expected answer rows (set-wise equality).
    Stores the winner back into state.SQL_meta_infos under this tool name.
    """

    def __init__(self):
        super().__init__()

    def _run(self, state: SystemState) -> Dict:
        if not state.mock_db_path:
            raise ValueError("Mock DB path is empty. Run mock_database_generator first.")

        # Collect candidates from previous generation step
        candidates = state.SQL_meta_infos.get("generate_candidate", [])
        if not candidates:
            # Fallback to use ground truth if present
            candidates = state.SQL_meta_infos.get("revise", []) or []

        expected = state.mock_expected_answer or {}
        expected_attrs: List[str] = expected.get("attributes", [])
        expected_values: List[Tuple] = [tuple(str(x) for x in row) for row in expected.get("values", [])]
        expected_set = set(expected_values)

        best_idx = None
        with sqlite3.connect(state.mock_db_path, timeout=30) as conn:
            cur = conn.cursor()
            for idx, meta in enumerate(candidates):
                sql = meta.SQL
                try:
                    cur.execute(sql)
                    rows = cur.fetchall()
                    result_set = set(tuple(str(x) for x in row[:len(expected_attrs)]) for row in rows)
                    if expected_attrs and result_set == expected_set:
                        best_idx = idx
                        break
                except Exception:
                    continue

        if best_idx is not None:
            state.SQL_meta_infos[self.tool_name] = [candidates[best_idx]]
        else:
            state.SQL_meta_infos[self.tool_name] = []

    def _get_updates(self, state: SystemState) -> Dict:
        chosen = state.SQL_meta_infos.get(self.tool_name, [])
        return {
            "node_type": self.tool_name,
            "selected_sql": chosen[0].SQL if chosen else "",
        }


