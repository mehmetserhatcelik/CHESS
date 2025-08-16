from workflow.agents.agent import Agent

from workflow.agents.reverse_tester.tool_kit.generate_reverse_question import GenerateReverseQuestion
from workflow.agents.reverse_tester.tool_kit.similarity_test import SimilarityTest


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
            "similarity_test": SimilarityTest(**config["tools"]["similarity_test"])            
        }


