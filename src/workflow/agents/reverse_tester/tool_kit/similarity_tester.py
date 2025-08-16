from typing import Dict, Optional

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.agents.tool import Tool
from workflow.sql_meta_info import SQLMetaInfo


class SimilarityTester(Tool):
    """Tool that selects the SQL whose generated question best matches the original question."""

    def __init__(self, template_name: str = None, engine_config: Dict = None, parser_name: str = None):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name
        self.SQL_id = None
        self.selected_index: Optional[int] = None

    def _run(self, state: SystemState):
        try:
            key = list(state.SQL_meta_infos.keys())[-1]
            candidates = state.SQL_meta_infos[key]
        except Exception as e:
            print(f"Error in SimilarityTester: {e}")
            return

        # Determine new key name for selected SQL
        if key.startswith(self.tool_name):
            idx = int(key[len(self.tool_name) + 1:])
            self.SQL_id = f"{self.tool_name}_{idx+1}"
        else:
            self.SQL_id = f"{self.tool_name}_1"
        state.SQL_meta_infos[self.SQL_id] = []

        formatted_questions = ""
        for i, sql_meta in enumerate(candidates):
            formatted_questions += f"{i+1}. {sql_meta.generated_question}\n"

        request_kwargs = {
            "QUESTION": state.task.question,
            "CANDIDATE_QUESTIONS": formatted_questions,
        }

        try:
            response = async_llm_chain_call(
                prompt=get_prompt(template_name=self.template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.parser_name),
                request_list=[request_kwargs],
                step=self.tool_name,
            )[0][0]
            index = response.get("index", 1)
        except Exception as e:
            print(f"Error selecting best question: {e}")
            index = 1
        self.selected_index = index
        chosen = candidates[index - 1] if candidates else SQLMetaInfo(SQL="SELECT 1")
        state.SQL_meta_infos[self.SQL_id].append(chosen)

    def _get_updates(self, state: SystemState) -> Dict:
        key = list(state.SQL_meta_infos.keys())[-2]
        candidates = state.SQL_meta_infos[key]
        return {
            "candidate_questions": [sql.generated_question for sql in candidates],
            "selected_index": self.selected_index,
            "selected_sql": state.SQL_meta_infos[self.SQL_id][0].SQL if state.SQL_meta_infos[self.SQL_id] else "",
        }
