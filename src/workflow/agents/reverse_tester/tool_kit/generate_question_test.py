from typing import Dict, List, Optional

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.sql_meta_info import SQLMetaInfo
from workflow.agents.tool import Tool


class GenerateQuestionTest(Tool):
    """
    LLM-based tool that generates discriminative tests for natural-language questions
    and selects the best reverse-generated question (thus the corresponding SQL)
    by evaluating which candidate passes the most tests.

    Acts as an alternative to `SimilarityTest`.
    """

    def __init__(
        self,
        generator_template_name: str = None,
        judge_template_name: str = None,
        engine_config: Dict = None,
        tests_parser_name: str = "generate_unit_tests",
        judge_parser_name: str = "similarity_judge",
        unit_test_count: int = 10,
        sampling_count: int = 1,
    ):
        super().__init__()
        self.generator_template_name = generator_template_name
        self.judge_template_name = judge_template_name
        self.engine_config = engine_config
        self.tests_parser_name = tests_parser_name
        self.judge_parser_name = judge_parser_name
        self.unit_test_count = unit_test_count
        self.sampling_count = sampling_count

        self.scores: List[float] = []
        self.comparison_matrix: List[List[int]] = []
        self.winner_index: int = -1
        self.SQL_id: Optional[str] = None
        self.generated_tests: List[str] = []

    def _run(self, state: SystemState):
        try:
            candidates_key = list(state.SQL_meta_infos.keys())[-1]
            candidates: List[SQLMetaInfo] = state.SQL_meta_infos[candidates_key]
            questions_key = list(state.reverse_questions.keys())[-1]
            questions: List[str] = state.reverse_questions[questions_key]
        except Exception as e:
            print(f"Error in GenerateQuestionTest: {e}")
            return

        # Deduplicate candidates to keep alignment with generated questions
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
            self._init_sql_bucket(state, candidates_key if candidates_key else self.tool_name)
            if candidates:
                state.SQL_meta_infos[self.SQL_id].append(candidates[0])
            return

        self._init_sql_bucket(state, candidates_key)

        # 1) Generate question tests with LLM
        tests_request = {
            "HINT": state.task.evidence,
            "INITIAL_QUESTION": state.task.question,
            "GENERATED_QUESTIONS": "\n".join([f"Question #{i+1}: {q}" for i, q in enumerate(questions)]),
            "UNIT_TEST_CAP": self.unit_test_count,
        }
        try:
            tests_responses = async_llm_chain_call(
                prompt=get_prompt(template_name=self.generator_template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.tests_parser_name),
                request_list=[tests_request],
                step=self.tool_name,
                sampling_count=self.sampling_count,
            )[0]
            self.generated_tests = []
            for resp in tests_responses:
                if isinstance(resp, dict) and "unit_tests" in resp:
                    self.generated_tests.extend(resp["unit_tests"]) 
        except Exception as e:
            print(f"Error generating question tests: {e}")
            self.generated_tests = []

        # Fallback: ensure we have at least a couple of basic tests
        if not self.generated_tests:
            self.generated_tests = [
                "The question should match the original intent and constraints.",
                "The question should mention the key entities referenced by the original question.",
            ]

        # 2) Judge per-test in separate requests and build comparison matrix
        request_list: List[Dict] = []
        for unit_test in self.generated_tests:
            try:
                judge_request = {
                    "HINT": state.task.evidence,
                    "INITIAL_QUESTION": state.task.question,
                    "GENERATED_QUESTIONS": "\n".join([f"Question #{i+1}: {q}" for i, q in enumerate(questions)]),
                    "QUESTION_TESTS": f"- {unit_test}",
                }
                request_list.append(judge_request)
            except Exception as e:
                print(f"Error creating judge request for question test: {e}")
                continue

        try:
            responses = async_llm_chain_call(
                prompt=get_prompt(template_name=self.judge_template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.judge_parser_name),
                request_list=request_list,
                step=self.tool_name,
            )
            responses = [r[0] for r in responses]
        except Exception as e:
            print(f"Error judging question tests: {e}")
            responses = []

        comparison_matrix: List[List[int]] = []
        for item in responses:
            try:
                scores = item.get("scores", []) if isinstance(item, dict) else []
                # ensure ints
                scores = [int(x) for x in scores]
                if scores:
                    comparison_matrix.append(scores)
            except Exception:
                continue

        # Fallback if parser did not return per-test scores
        if not comparison_matrix:
            # Try single-shot judging as fallback
            tests_block = "\n".join([f"- {t}" for t in self.generated_tests])
            fallback_request = {
                "HINT": state.task.evidence,
                "INITIAL_QUESTION": state.task.question,
                "GENERATED_QUESTIONS": "\n".join([f"Question #{i+1}: {q}" for i, q in enumerate(questions)]),
                "QUESTION_TESTS": tests_block,
            }
            try:
                judge_response = async_llm_chain_call(
                    prompt=get_prompt(template_name=self.judge_template_name),
                    engine=get_llm_chain(**self.engine_config),
                    parser=get_parser(self.judge_parser_name),
                    request_list=[fallback_request],
                    step=self.tool_name,
                )
                result = judge_response[0][0] if judge_response and judge_response[0] else None
                if isinstance(result, dict) and isinstance(result.get("scores", None), list):
                    one_row = [int(x) for x in result["scores"]]
                    comparison_matrix = [one_row]
            except Exception:
                pass

        self.comparison_matrix = comparison_matrix

        if not comparison_matrix:
            # default to first candidate
            self.scores = [1] + [0] * (len(candidates) - 1)
            self.winner_index = 0
            state.SQL_meta_infos[self.SQL_id].append(candidates[0])
            state.unit_tests["question_test_generation"] = list(self.generated_tests)
            return

        # Sum scores across tests per candidate
        try:
            aggregated_scores = [
                sum(row[idx] for row in comparison_matrix)
                for idx in range(len(comparison_matrix[0]))
            ]
        except Exception:
            aggregated_scores = []

        self.scores = aggregated_scores

        # Tie-break using execution-based clustering like UnitTester
        candidates_clusters = self.execution_based_clustering(candidates)
        best_candidate = self.pick_the_best_candidate(aggregated_scores, candidates, candidates_clusters)
        try:
            self.winner_index = candidates.index(best_candidate)
        except Exception:
            self.winner_index = 0
            best_candidate = candidates[0]

        # Select corresponding SQL
        state.SQL_meta_infos[self.SQL_id].append(best_candidate)

        # Store tests for visibility
        state.unit_tests["question_test_generation"] = list(self.generated_tests)

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
            "tests": list(self.generated_tests),
            "scores": self.scores,
            "comparison_matrix": self.comparison_matrix,
            "winner_index": self.winner_index,
            "candidates": [sql_meta_info.SQL for sql_meta_info in target_SQL_meta_infos],
            "selected_candidate": state.SQL_meta_infos[self.SQL_id][0].SQL,
        }

    def execution_based_clustering(self, candidate_queries: List[SQLMetaInfo]) -> Dict:
        clusters: Dict[str, List[SQLMetaInfo]] = {}
        for query in candidate_queries:
            try:
                result = str(query.execution_result) if isinstance(query.execution_result, str) else repr(query.execution_result)
            except Exception:
                continue
            if result not in clusters:
                clusters[result] = []
            clusters[result].append(query)
        return clusters

    def pick_the_best_candidate(self, scores: List[int], candidates: List[SQLMetaInfo], candidate_clusters: Dict) -> SQLMetaInfo:
        if not candidates:
            raise ValueError("No candidates to pick from")
        if not scores or len(scores) != len(candidates):
            return candidates[0]
        try:
            max_score = max(scores)
            best_candidates = [candidates[index] for index, score in enumerate(scores) if score == max_score]
            if len(best_candidates) == 1:
                return best_candidates[0]
            # tie-break by largest execution cluster
            largest_cluster = max(candidate_clusters, key=lambda x: len(candidate_clusters[x])) if candidate_clusters else None
            if largest_cluster is not None:
                for candidate in best_candidates:
                    if candidate in candidate_clusters[largest_cluster]:
                        return candidate
            return best_candidates[0]
        except Exception:
            return candidates[0]


