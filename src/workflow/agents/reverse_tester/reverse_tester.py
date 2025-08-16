from workflow.agents.agent import Agent

from workflow.agents.reverse_tester.tool_kit.question_generator import QuestionGenerator
from workflow.agents.reverse_tester.tool_kit.similarity_tester import SimilarityTester


class ReverseTester(Agent):
    """Agent responsible for generating questions from SQL and selecting the most similar to the user question."""

    def __init__(self, config: dict):
        super().__init__(
            name="reverse_tester",
            task=("generate natural language questions for each candidate SQL and select the query whose question best matches the original question"),
            config=config,
        )

        self.tools = {
            "question_generator": QuestionGenerator(**config["tools"]["question_generator"]),
            "similarity_tester": SimilarityTester(**config["tools"]["similarity_tester"]),
        }
