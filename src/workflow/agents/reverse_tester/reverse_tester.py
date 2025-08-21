from workflow.agents.agent import Agent

from workflow.agents.reverse_tester.tool_kit.generate_reverse_question import GenerateReverseQuestion
from workflow.agents.reverse_tester.tool_kit.similarity_test import SimilarityTest
from workflow.agents.reverse_tester.tool_kit.generate_question_test import GenerateQuestionTest


class ReverseTester(Agent):
    """
    Agent responsible for generating natural language questions from candidate SQLs
    and selecting the SQL whose generated question is most similar to the initial question.
    """
    
    def __init__(self, config: dict):
        super().__init__(
            name="reverse_tester",
            task=(
                "generate a natural language question for each candidate SQL, then judge which generated question is most similar to the initial question",
                "pick the corresponding SQL as the final candidate"
            ),
            config=config,
        )

        self.tools = {
            "generate_reverse_question": GenerateReverseQuestion(**config["tools"]["generate_reverse_question"]),
            "similarity_test": SimilarityTest(**config["tools"].get("similarity_test", {})),
            "generate_question_test": GenerateQuestionTest(**config["tools"].get("generate_question_test", {}))
        }


    def workout(self, system_state):
        """
        Deterministic workflow for reverse testing:
        1) Generate reverse questions for the latest SQL candidates
        2) Run similarity_test to pick the most similar question to the user's initial question
        Returns the updated system state.
        """
        # Step 1: generate reverse questions
        generate_tool = self.tools["generate_reverse_question"]
        generate_tool(system_state)

        # Step 2: pick the best matching question -> corresponding SQL
        if self.config["tools"].get("generate_question_test"):
            self.tools["generate_question_test"](system_state)
        else:
            self.tools["similarity_test"](system_state)

        return system_state
