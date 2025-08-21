import json
import re
import logging
from ast import literal_eval
from typing import Any, Dict, List, Tuple

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.exceptions import OutputParserException

class PythonListOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing Python lists."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Any:
        """
        Parses the output to extract Python list content from markdown.

        Args:
            output (str): The output string containing Python list.

        Returns:
            Any: The parsed Python list.
        """
        logging.debug(f"Parsing output with PythonListOutputParser: {output}")
        if "```python" in output:
            output = output.split("```python")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        return eval(output)  # Note: Using eval is potentially unsafe, consider using ast.literal_eval if possible.

class FilterColumnOutput(BaseModel):
    """Model for filter column output."""
    chain_of_thought_reasoning: str = Field(description="One line explanation of why or why not the column information is relevant to the question and the hint.")
    is_column_information_relevant: str = Field(description="Yes or No")

class SelectTablesOutputParser(BaseOutputParser):
    """Parses select tables outputs embedded in markdown code blocks containing JSON."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Any:
        """
        Parses the output to extract JSON content from markdown.

        Args:
            output (str): The output string containing JSON.

        Returns:
            Any: The parsed JSON content.
        """
        logging.debug(f"Parsing output with SelectTablesOutputParser: {output}")
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        output = output.replace("\n", " ").replace("\t", " ")
        return json.loads(output)

class ColumnSelectionOutput(BaseModel):
    """Model for column selection output."""
    table_columns: Dict[str, Tuple[str, List[str]]] = Field(description="A mapping of table and column names to a tuple containing the reason for the column's selection and a list of keywords for data lookup. If no keywords are required, an empty list is provided.")

class GenerateCandidateOutput(BaseModel):
    """Model for SQL generation output."""
    chain_of_thought_reasoning: str = Field(description="Your thought process on how you arrived at the final SQL query.")
    SQL: str = Field(description="The generated SQL query in a single string.")

class GenerateCandidateFinetunedMarkDownParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "```sql" in output:
            output = output.split("```sql")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        return {"SQL": output}
    
class ReviseOutput(BaseModel):
    """Model for SQL revision output."""
    chain_of_thought_reasoning: str = Field(description="Your thought process on how you arrived at the final SQL query.")
    revised_SQL: str = Field(description="The revised SQL query in a single string.")

    
class GenerateCandidateGeminiMarkDownParserCOT(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with RecapOutputParserCOT: {output}")
        plan = ""
        if "<FINAL_ANSWER>" in output and "</FINAL_ANSWER>" in output:
            plan = output.split("<FINAL_ANSWER>")[0]
            output = output.split("<FINAL_ANSWER>")[1].split(
            "</FINAL_ANSWER>"
            )[0]
        query = output.replace("```sql", "").replace("```", "").replace("\n", " ")
        return {"SQL": query, "plan": plan}
    
class GeminiMarkDownOutputParserCOT(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParserCoT: {output}")
        if "My final answer is:" in output:
            plan, query = output.split("My final answer is:")
        else:
            plan, query = output, output
        if "```sql" in query:
            query = query.split("```sql")[1].split("```")[0]
        query = re.sub(r"^\s+", "", query)
        return {"SQL": query, "plan": plan}

class ReviseGeminiOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with CheckerOutputParser: {output}")
        if "<FINAL_ANSWER>" in output and "</FINAL_ANSWER>" in output:
            output = output.split("<FINAL_ANSWER>")[1].split(
            "</FINAL_ANSWER>"
            )[0]
        if "<FINAL_ANSWER>" in output:
            output = output.split("<FINAL_ANSWER>")[1]
        query = output.replace("```sql", "").replace("```", "").replace("\n", " ")
        return {"refined_sql_query": query}

   
class ListOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output a list

        Args:
            output (str): A string containing a list.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        try:
            output = literal_eval(output)
        except Exception as e:
            raise OutputParserException(f"Error parsing list: {e}")
        return output
    

class UnitTestEvaluationOutput(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "<Answer>" in output and "</Answer>" in output:
            output = output.split("<Answer>")[1].split(
            "</Answer>"
            )[0].strip()
        else:
            raise OutputParserException("Your answer is not in the correct format. Please make sure to include your answer in the format <Answer>...</Answer>")
        scores = []
        for line in output.split("\n"):
            if ":" in line:
                try:
                    key, value = line.split(":")
                    if "passed" in value.lower():
                        scores.append(1)
                    else:
                        scores.append(0)
                except Exception as e:
                    raise OutputParserException(f"Error parsing unit test evaluation: {e}, each line should be in the format 'unit test #n: Passed/Failed'")
        return {"scores": scores}
    
class TestCaseGenerationOutput(BaseOutputParser):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "<Answer>" in output and "</Answer>" in output:
            output = output.split("<Answer>")[1].split(
            "</Answer>"
            )[0]
        else:
            raise OutputParserException("Your answer is not in the correct format. Please make sure to include your answer in the format <Answer>...</Answer>")
        try:
            unit_tests = literal_eval(output)
        except Exception as e:
            raise OutputParserException(f"Error parsing test case generation: {e}")
        return {"unit_tests": unit_tests}

class PlainTextOrJSONQuestionParser(BaseOutputParser):
    """Parses either plain text question or a JSON object {"question": str}."""
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
    def parse(self, output: str) -> Any:
        try:
            txt = output.strip()
            # Try parse a minimal JSON first
            if txt.startswith("{") and txt.endswith("}"):
                data = json.loads(txt)
                if isinstance(data, dict) and "question" in data:
                    return {"question": str(data["question"]).strip()}
            # Otherwise, treat as plain text answer possibly within tags
            if "```" in txt:
                # take first fenced block content
                txt = txt.split("```")[1]
            # remove tag wrappers if present
            if "<Answer>" in txt and "</Answer>" in txt:
                txt = txt.split("<Answer>")[1].split("</Answer>")[0]
            return {"question": txt.strip()}
        except Exception as e:
            raise OutputParserException(f"Error parsing reverse question: {e}")

class ESQLQuestionEnrichmentOutput(BaseModel):
    chain_of_thought_reasoning: str = Field(description="Detail explanation of the logical analysis that led to the refined question, considering detailed possible sql generation steps")
    enriched_question: str = Field(description="Expanded and refined question which is more understandable, clear and free of irrelevant information.")

class ESQLQuestionEnrichmentParser(BaseOutputParser):
    """Parses E-SQL style JSON with keys chain_of_thought_reasoning and enriched_question from fenced block or raw text."""
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Any:
        text = output or ""
        text = str(text)
        # Try JSON first
        try:
            candidate = text
            if "```json" in candidate:
                candidate = candidate.split("```json")[1].split("```")[0]
            candidate = candidate.replace("\n", " ").replace("\t", " ").strip()
            data = JsonOutputParser(pydantic_object=ESQLQuestionEnrichmentOutput).parse(candidate)
            return {"question": data.enriched_question, "_cot": data.chain_of_thought_reasoning}
        except Exception:
            pass
        # Try to extract <Answer> block as plain text
        try:
            txt = text.strip()
            if "<Answer>" in txt and "</Answer>" in txt:
                ans = txt.split("<Answer>")[1].split("</Answer>")[0].strip()
                return {"question": ans}
            # Fallback to first fenced block
            if "```" in txt:
                ans = txt.split("```")[1].strip()
                return {"question": ans}
            # Fallback to full text
            return {"question": txt.strip()}
        except Exception as e:
            raise OutputParserException(f"Error parsing E-SQL enrichment output: {e}")

class SimilarityJudgeParser(BaseOutputParser):
    """Parses judge output containing winner_index and optional scores list."""
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
    def parse(self, output: str) -> Any:
        try:
            text = output
            if "<Answer>" in text and "</Answer>" in text:
                text = text.split("<Answer>")[1].split("</Answer>")[0]
            # normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            # extract winner_index
            m = re.search(r"winner_index\s*:\s*(\d+)", text, flags=re.IGNORECASE)
            if not m:
                raise OutputParserException("winner_index not found")
            winner_index_1_based = int(m.group(1))
            winner_index = max(0, winner_index_1_based - 1)
            # optional scores list
            scores = None
            m2 = re.search(r"scores\s*:\s*(\[[^\]]*\])", text, flags=re.IGNORECASE)
            if m2:
                try:
                    scores = literal_eval(m2.group(1))
                except Exception:
                    scores = None
            result = {"winner_index": winner_index}
            if isinstance(scores, list):
                result["scores"] = scores
            return result
        except OutputParserException:
            raise
        except Exception as e:
            raise OutputParserException(f"Error parsing similarity judge output: {e}")

def get_parser(parser_name: str) -> BaseOutputParser:
    """
    Returns the appropriate parser based on the provided parser name.

    Args:
        parser_name (str): The name of the parser to retrieve.

    Returns:
        BaseOutputParser: The appropriate parser instance.

    Raises:
        ValueError: If the parser name is invalid.
    """
    parser_configs = {
        "python_list_output_parser": PythonListOutputParser,
        "filter_column": lambda: JsonOutputParser(pydantic_object=FilterColumnOutput),
        "select_tables": lambda: JsonOutputParser(pydantic_object=SelectTablesOutputParser),
        "select_columns": lambda: JsonOutputParser(pydantic_object=ColumnSelectionOutput),
        "generate_candidate": lambda: JsonOutputParser(pydantic_object=GenerateCandidateOutput),
        "generated_candidate_finetuned": GenerateCandidateFinetunedMarkDownParser(),
        "revise": lambda: JsonOutputParser(pydantic_object=ReviseOutput),
        "generate_candidate_gemini_markdown_cot": GenerateCandidateGeminiMarkDownParserCOT(),
        "generate_candidate_gemini_cot": GeminiMarkDownOutputParserCOT(),
        "revise_new": ReviseGeminiOutputParser(),
        "list_output_parser": ListOutputParser(),
        "evaluate": UnitTestEvaluationOutput(),
        "generate_unit_tests": TestCaseGenerationOutput(),
        "reverse_question": PlainTextOrJSONQuestionParser(),
        "similarity_judge": SimilarityJudgeParser(),
        "esql_question_enrichment": ESQLQuestionEnrichmentParser()
    }

    if parser_name not in parser_configs:
        logging.error(f"Invalid parser name: {parser_name}")
        raise ValueError(f"Invalid parser name: {parser_name}")

    logging.info(f"Retrieving parser for: {parser_name}")
    parser = parser_configs[parser_name]() if callable(parser_configs[parser_name]) else parser_configs[parser_name]
    return parser
