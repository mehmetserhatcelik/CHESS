from workflow.agents.agent import Agent

from workflow.agents.mock_db_tester.tool_kit.mock_database_generator import MockDatabaseGenerator
from workflow.agents.mock_db_tester.tool_kit.mock_answer_generator import MockAnswerGenerator
from workflow.agents.mock_db_tester.tool_kit.mock_sql_decision import MockSQLDecision


class MockDBTester(Agent):
    """
    Agent that creates a small mock SQLite database consistent with the schema,
    inserts satisfying and distractor rows, derives the expected answer table,
    executes candidate SQLs on the mock DB, and picks the SQL that matches the expected answer.
    """
    
    def __init__(self, config: dict):
        super().__init__(
            name="mock_db_tester",
            task=(
                "Generate a minimal mock database with a few rows satisfying the question criteria and more non-satisfying rows; "
                "derive the expected answer and select the candidate SQL whose execution matches it.",
            ),
            config=config,
        )

        self.tools = {
            "mock_database_generator": MockDatabaseGenerator(**config["tools"].get("mock_database_generator", {})),
            "mock_answer_generator": MockAnswerGenerator(**config["tools"].get("mock_answer_generator", {})),
            "mock_sql_decision": MockSQLDecision(**config["tools"].get("mock_sql_decision", {})),
        }

    def workout(self, system_state):
        # Deterministic sequence for mock DB testing
        self.tools["mock_database_generator"](system_state)
        self.tools["mock_answer_generator"](system_state)
        self.tools["mock_sql_decision"](system_state)
        return system_state


