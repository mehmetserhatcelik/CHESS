from typing import Dict
import os

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
            # Load and format few-shot examples (E-SQL style) if available
            fewshot_examples = ""
            try:
                fewshot_path = os.path.join("templates", "fewshot_question_enrichment_examples.json")
                if os.path.exists(fewshot_path):
                    import json
                    with open(fewshot_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    levels = ["simple", "moderate", "challanging"]
                    selected_blocks = []
                    for level in levels:
                        examples = data.get(level, [])
                        # prefer examples from different db than current
                        filtered = [ex for ex in examples if ex.get("db_id") != state.task.db_id]
                        pool = filtered if filtered else examples
                        if not pool:
                            continue
                        ex = pool[0]
                        block = []
                        block.append(f"Question: {ex.get('question','')}")
                        block.append(f"Evidence: {ex.get('evidence','')}")
                        if ex.get('enrichment_reasoning'):
                            block.append(f"Enrichment Reasoning: {ex.get('enrichment_reasoning')}")
                        # prefer v2 if present
                        enriched = ex.get('question_enriched_v2') or ex.get('question_enriched') or ""
                        block.append(f"Enriched Question: {enriched}")
                        selected_blocks.append("\n".join(block))
                    fewshot_examples = "\n\n".join(selected_blocks)
            except Exception as e:
                print(f"Error loading few-shot examples: {e}")
            request_kwargs = {
                # E-SQL style variables
                "SCHEMA": database_schema,
                "DB_DESCRIPTIONS": state.get_schema_string(schema_type="complete", include_value_description=True),
                "DB_SAMPLES": state.get_schema_string(schema_type="complete", include_value_description=False),
                "POSSIBLE_CONDITIONS": "",
                "QUESTION": state.task.question,
                "EVIDENCE": state.task.evidence,
                "FEWSHOT_EXAMPLES": fewshot_examples,
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
                state.enriched_initial_question = result["question"].strip()
            elif isinstance(result, str):
                state.enriched_initial_question = result.strip()
        except Exception as e:
            print(f"Error storing enriched initial question: {e}")

    def _get_updates(self, state: SystemState) -> Dict:
        return {
            "enriched_initial_question": state.enriched_initial_question
        }


