from typing import Dict, List

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.agents.tool import Tool


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
                request_kwargs = {
                    "DATABASE_SCHEMA": database_schema,
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

    def _get_updates(self, state: SystemState) -> Dict:
        key = list(state.reverse_questions.keys())[-1]
        return {
            "node_type": self.tool_name,
            "questions": state.reverse_questions[key]
        }


