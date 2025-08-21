from typing import Dict

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.agents.tool import Tool


class EnrichInitialQuestion(Tool):
    """
    Enriches the initial NL question with explicit intent, constraints, joins, and aggregates.
    Output becomes a more structured, precise question description for similarity anchoring.
    """

    def __init__(self, template_name: str = None, engine_config: Dict = None, parser_name: str = "esql_question_enrichment"):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name

    def _run(self, state: SystemState):
        try:
            database_schema = state.get_schema_string(schema_type="complete", include_value_description=True)
            request_kwargs = {
                # E-SQL style variables
                "SCHEMA": database_schema,
                "DB_DESCRIPTIONS": state.get_schema_string(schema_type="complete", include_value_description=True),
                "DB_SAMPLES": "",
                "POSSIBLE_CONDITIONS": "",
                "QUESTION": state.task.question,
                "EVIDENCE": state.task.evidence,
                "FEWSHOT_EXAMPLES": "",
            }
        except Exception as e:
            print(f"Error preparing EnrichInitialQuestion request: {e}")
            return

        try:
            response = async_llm_chain_call(
                prompt=get_prompt(template_name=self.template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.parser_name),
                request_list=[request_kwargs],
                step=self.tool_name,
            )
            result = response[0][0] if response and response[0] else None
        except Exception as e:
            print(f"Error in EnrichInitialQuestion call: {e}")
            result = None

        try:
            if isinstance(result, dict) and "question" in result:
                state.enriched_initial_question = result["question"]
            elif isinstance(result, str):
                state.enriched_initial_question = result
        except Exception as e:
            print(f"Error storing enriched initial question: {e}")

    def _get_updates(self, state: SystemState) -> Dict:
        return {
            "enriched_initial_question": state.enriched_initial_question
        }


