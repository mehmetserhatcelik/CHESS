from typing import Dict, List, Optional
import json
import os

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.agents.tool import Tool
from runner.database_manager import DatabaseManager


class GenerateReverseQuestion(Tool):
    """
    Tool for generating natural language questions from each candidate SQL.
    """

    def __init__(self, template_name: str = None, engine_config: Dict = None, parser_name: str = None, sampling_count: int = 1):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name
        self.sampling_count = sampling_count
        self._column_meanings_cache: Optional[Dict] = None

    def _run(self, state: SystemState):
        try:
            # Pick the last generated candidate list
            key_to_reverse = list(state.SQL_meta_infos.keys())[-1]
            target_SQL_meta_infos = state.SQL_meta_infos[key_to_reverse]
        except Exception as e:
            print(f"Error in GenerateReverseQuestion: {e}")
            return

        # Create a new slot for questions produced by this tool call
        if key_to_reverse.startswith(self.tool_name):
            id = int(key_to_reverse[len(self.tool_name)+1:])
            questions_id = self.tool_name + "_" + str(id+1)
        else:
            questions_id = self.tool_name + "_1"
        state.reverse_questions[questions_id] = []

        if not target_SQL_meta_infos:
            return

        request_list = []
        database_schema = state.get_database_schema_for_queries([sql_meta_info.SQL for sql_meta_info in target_SQL_meta_infos])
        for sql_meta_info in target_SQL_meta_infos:
            try:
                referenced_columns = self._get_referenced_columns(sql_meta_info.SQL)
                column_meanings_text = self._build_column_meanings_text(state.task.db_id, referenced_columns)
                request_kwargs = {
                    "DATABASE_SCHEMA": database_schema,
                    "INITIAL_QUESTION": state.task.question,
                    "COLUMN_MEANINGS": column_meanings_text,
                    "SQL": sql_meta_info.SQL,
                }
                request_list.append(request_kwargs)
            except Exception as e:
                print(f"Error creating request kwargs for GenerateReverseQuestion: {e}")
                continue

        try:
            response = async_llm_chain_call(
                prompt=get_prompt(template_name=self.template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.parser_name),
                request_list=request_list,
                step=self.tool_name,
                sampling_count=self.sampling_count
            )
            # Flatten samples per SQL; take first sample per input as the representative question
            response = [res[0] for res in response if res]
        except Exception as e:
            print(f"Error generating reverse questions: {e}")
            response = []

        for res in response:
            try:
                # Parser returns {"question": str} or a raw string; normalize to string
                if isinstance(res, dict) and "question" in res:
                    question_text = res["question"]
                else:
                    question_text = str(res)
                state.reverse_questions[questions_id].append(question_text)
            except Exception as e:
                print(f"Error appending reverse question: {e}")
                continue

    def _get_referenced_columns(self, sql: str) -> Dict[str, List[str]]:
        """
        Uses DatabaseManager to extract a mapping of referenced columns per table.
        """
        try:
            columns_dict = DatabaseManager().get_sql_columns_dict(sql=sql)
            return columns_dict or {}
        except Exception as e:
            print(f"Error extracting referenced columns: {e}")
            return {}

    def _load_column_meanings(self, db_id: str) -> Optional[Dict]:
        """
        Loads column meanings JSON for the given database id. Looks for:
        - src/workflow/agents/reverse_tester/tool_kit/column_meanings_{db_id}.json
        - fallback: src/workflow/agents/reverse_tester/tool_kit/column_meanings_california_schools.json
        Returns the parsed dict or None.
        """
        if self._column_meanings_cache is not None:
            return self._column_meanings_cache
        base_dir = os.path.join("src", "workflow", "agents", "reverse_tester", "tool_kit")
        candidates = [
            os.path.join(base_dir, f"column_meanings_{db_id}.json"),
            os.path.join(base_dir, "column_meanings_california_schools.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._column_meanings_cache = data
                        return data
                except Exception as e:
                    print(f"Error loading column meanings from {path}: {e}")
                    continue
        return None

    def _build_column_meanings_text(self, db_id: str, referenced_columns: Dict[str, List[str]]) -> str:
        """
        Builds a human-readable meanings block for only the referenced columns.
        """
        data = self._load_column_meanings(db_id)
        if not data or db_id not in data:
            # Try single-root layout like {"california_schools": {...}}
            root_key = next(iter(data.keys())) if isinstance(data, dict) and data else None
            db_block = data.get(root_key, {}) if root_key else {}
        else:
            db_block = data.get(db_id, {})

        lines: List[str] = []
        for table_name, columns in referenced_columns.items():
            table_block = db_block.get(table_name, {}) if isinstance(db_block, dict) else {}
            for col in columns:
                meaning = table_block.get(col)
                if meaning:
                    # Strip leading comment markers if present
                    meaning = str(meaning).lstrip("#").strip()
                    lines.append(f"- {table_name}.{col}: {meaning}")
        return "\n".join(lines) if lines else "(No additional meanings found)"

    def _get_updates(self, state: SystemState) -> Dict:
        key = list(state.reverse_questions.keys())[-1]
        return {
            "node_type": self.tool_name,
            "questions": state.reverse_questions[key]
        }


