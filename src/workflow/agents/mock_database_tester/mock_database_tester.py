from workflow.agents.agent import Agent

from .tool_kit.mock_database_generator import MockDatabaseGenerator
from .tool_kit.mock_answer_generator import MockAnswerGenerator
from .tool_kit.decision import Decision


class MockDatabaseTester(Agent):
    """Agent that builds a mock database and selects the correct SQL."""

    def __init__(self, config: dict):
        super().__init__(
            name="Mock Database Tester",
            task=(
                "create a mock database, derive the expected answer, and choose the SQL that matches the answer",
                "generate a mock database, compute the expected answer, and pick the candidate SQL that returns it",
            ),
            config=config,
        )
        self.tools = {
            "mock_database_generator": MockDatabaseGenerator(**config["tools"]["mock_database_generator"]),
            "mock_answer_generator": MockAnswerGenerator(**config["tools"]["mock_answer_generator"]),
            "decision": Decision(),
        }
