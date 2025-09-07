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
            # Execute DDL first, then DML
            ddl_statements: List[str] = []
            dml_statements: List[str] = []
            for stmt in self.generated_sql:
                if not isinstance(stmt, str):
                    continue
                sql = stmt.strip().rstrip(";")
                if not sql:
                    continue
                upper = sql.upper()
                if upper.startswith("CREATE TABLE") or upper.startswith("DROP TABLE"):
                    ddl_statements.append(sql)
                elif upper.startswith("INSERT INTO"):
                    dml_statements.append(sql)
                else:
                    # Ignore other statements for safety in mock DB setup
                    continue

            # Run DDL
            for ddl in ddl_statements:
                cursor.execute(ddl)

            # Helper to ensure table exists before INSERT
            def ensure_table_exists_for_insert(insert_sql: str) -> None:
                try:
                    # Extract table name and values count to build a simple TEXT schema if missing
                    after_into = insert_sql.split("INTO", 1)[1].strip()
                    table_and_rest = after_into.split(None, 1)
                    table_ident = table_and_rest[0].strip()
                    # Remove optional backticks/quotes
                    table_clean = table_ident.strip("`\"")
                    # Check if table exists
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_clean,))
                    if cursor.fetchone():
                        return
                    # Infer column count from VALUES (...)
                    values_part = insert_sql.upper().split("VALUES", 1)[1]
                    first_tuple = values_part.split("(", 1)[1].split(")", 1)[0]
                    num_cols = len([c for c in first_tuple.split(",")])
                    cols = ", ".join([f"col{i+1} TEXT" for i in range(max(1, num_cols))])
                    cursor.execute(f"CREATE TABLE IF NOT EXISTS `{table_clean}` ({cols})")
                except Exception:
                    # Best-effort; if parsing fails, let the INSERT raise naturally
                    pass

            # Run DML with best-effort auto-create
            for dml in dml_statements:
                ensure_table_exists_for_insert(dml)
                cursor.execute(dml)
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
