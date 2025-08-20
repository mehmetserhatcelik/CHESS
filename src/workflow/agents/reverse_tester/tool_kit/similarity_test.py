from typing import Dict, List, Optional

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from llm.embeddings import (
    get_embedding_client,
    embed_texts,
    compute_pairwise_similarities,
)
from workflow.system_state import SystemState
from workflow.sql_meta_info import SQLMetaInfo
from workflow.agents.tool import Tool


class SimilarityTest(Tool):
    """
    Tool that asks an LLM to judge which generated question is most similar to the initial question.
    Selects the corresponding SQL accordingly.
    """

    def __init__(self, template_name: str = None, engine_config: Dict = None, parser_name: str = None, embedding_config: Optional[Dict] = None, mode: Optional[str] = None):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name
        self.embedding_config = embedding_config or {}
        self.mode = 'embeddings'
        self.scores: List[float] = []
        self.winner_index: int = -1
        self.SQL_id: str = None

    def _run(self, state: SystemState):
        try:
            candidates_key = list(state.SQL_meta_infos.keys())[-1]
            candidates: List[SQLMetaInfo] = state.SQL_meta_infos[candidates_key]
            questions_key = list(state.reverse_questions.keys())[-1]
            questions: List[str] = state.reverse_questions[questions_key]
        except Exception as e:
            print(f"Error in SimilarityTest: {e}")
            return

        # Remove duplicates from candidates to match questions (same logic as generate_reverse_question)
        if candidates:
            seen_sqls = set()
            unique_candidates = []
            for sql_meta_info in candidates:
                sql_text = sql_meta_info.SQL
                if sql_text not in seen_sqls:
                    unique_candidates.append(sql_meta_info)
                    seen_sqls.add(sql_text)
            candidates = unique_candidates

        if not candidates or not questions or len(candidates) != len(questions):
            self._init_sql_bucket(state, candidates_key)
            if candidates:
                state.SQL_meta_infos[self.SQL_id].append(candidates[0])
            return

        self._init_sql_bucket(state, candidates_key)

        used_embeddings = False
        if self.mode in ("embeddings", "auto"):
            try:
                # Default to Vertex AI embeddings unless overridden in YAML
                model_name = self.embedding_config.get("model", "text-embedding-004")
                provider = self.embedding_config.get("provider", "vertexai")
                device = self.embedding_config.get("device")
                pooling = self.embedding_config.get("pooling")
                max_length = self.embedding_config.get("max_length")

                client = get_embedding_client(
                    model=model_name,
                    provider=provider,
                    device=device,
                    pooling=pooling,
                    max_length=max_length,
                )

                anchor_text = state.task.question
                candidate_texts = questions
                vectors = embed_texts(client, [anchor_text] + candidate_texts)
                anchor_vec, candidate_vecs = vectors[0], vectors[1:]
                sims = compute_pairwise_similarities(anchor_vec, candidate_vecs)
                self.scores = sims
                if len(sims) > 0:
                    self.winner_index = max(range(len(sims)), key=lambda i: sims[i])
                    used_embeddings = True
            except Exception as e:
                print(f"Error in SimilarityTest embeddings flow: {e}")
                used_embeddings = False

        if not used_embeddings:
            request_kwargs = {
                "INITIAL_QUESTION": state.task.question,
                "GENERATED_QUESTIONS": "\n".join([f"Question #{i+1}: {q}" for i, q in enumerate(questions)]),
            }

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
                print(f"Error in SimilarityTest calling LLM: {e}")
                result = None

            if isinstance(result, dict):
                self.winner_index = int(result.get("winner_index", -1))
                if "scores" in result and isinstance(result["scores"], list):
                    try:
                        self.scores = [float(x) for x in result["scores"]]
                    except Exception:
                        self.scores = []
            else:
                self.winner_index = -1

        if self.winner_index is None or self.winner_index < 0 or self.winner_index >= len(questions):
            self.winner_index = 0

        # winner_index is 0-based index into questions list, use it to select corresponding SQL
        state.SQL_meta_infos[self.SQL_id].append(candidates[self.winner_index])

    def _init_sql_bucket(self, state: SystemState, base_key: str):
        if base_key.startswith(self.tool_name):
            id = int(base_key[len(self.tool_name)+1:])
            self.SQL_id = self.tool_name + "_" + str(id+1)
        else:
            self.SQL_id = self.tool_name + "_1"
        state.SQL_meta_infos[self.SQL_id] = []

    def _get_updates(self, state: SystemState) -> Dict:
        key_to_evaluate = list(state.SQL_meta_infos.keys())[-2]
        target_SQL_meta_infos = state.SQL_meta_infos[key_to_evaluate]
        return {
            "scores": self.scores,
            "winner_index": self.winner_index,
            "candidates": [sql_meta_info.SQL for sql_meta_info in target_SQL_meta_infos],
            "selected_candidate": state.SQL_meta_infos[self.SQL_id][0].SQL
        }


