from typing import Dict, List

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.agents.tool import Tool


class QuestionGenerator(Tool):
    """Tool to generate natural language questions for candidate SQL queries."""

    def __init__(self, template_name: str = None, engine_config: Dict = None, parser_name: str = None, sampling_count: int = 1):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name
        self.sampling_count = sampling_count
        self.generated_questions: List[str] = []

    def _run(self, state: SystemState):
        try:
            key = list(state.SQL_meta_infos.keys())[-1]
            target_sqls = state.SQL_meta_infos[key]
        except Exception as e:
            print(f"Error in QuestionGenerator: {e}")
            return

        request_list = []
        for sql_meta in target_sqls:
            try:
                database_schema = state.get_database_schema_for_queries([sql_meta.SQL])
                request_kwargs = {
                    "DATABASE_SCHEMA": database_schema,
                    "SQL": sql_meta.SQL,
                }
                request_list.append(request_kwargs)
            except Exception as e:
                print(f"Error preparing request for question generation: {e}")
                request_list.append({})

        try:
            responses = async_llm_chain_call(
                prompt=get_prompt(template_name=self.template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.parser_name),
                request_list=request_list,
                step=self.tool_name,
                sampling_count=self.sampling_count,
            )
            responses = [r[0] for r in responses]
        except Exception as e:
            print(f"Error generating questions: {e}")
            responses = []

        self.generated_questions = []
        for sql_meta, res in zip(target_sqls, responses):
            question = res.get("question", "") if isinstance(res, dict) else ""
            sql_meta.generated_question = question
            self.generated_questions.append(question)

    def _get_updates(self, state: SystemState) -> Dict:
        key = list(state.SQL_meta_infos.keys())[-1]
        target_sqls = state.SQL_meta_infos[key]
        return {
            "generated_questions": [
                {"SQL": sql_meta.SQL, "question": sql_meta.generated_question}
                for sql_meta in target_sqls
            ]
        }
