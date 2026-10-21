from typing import List
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    """The model's critique of its own answer."""

    missing: str = Field(
        description="ONE sentence: the most important fact the answer is missing."
    )
    superfluous: str = Field(
        description="ONE sentence: what in the answer is unnecessary. Write 'nothing' if there is none."
    )


class AnswerQuestion(BaseModel):
    """First response: answer, self-critique, and what to search for."""

    answer: str = Field(
        description="Plain-text answer in 2 to 4 sentences. No emojis, no markdown."
    )
    reflection: Reflection = Field(description="A short critique of your own answer.")
    search_queries: List[str] = Field(
        description="1 to 3 web search queries written as keywords, e.g. 'LangGraph checkpointer persistence'."
    )


class ReviseAnswer(AnswerQuestion):
    """Revised response: everything in AnswerQuestion, plus sources."""

    references: List[str] = Field(
        description="URLs of the sources used. Only URLs that appeared in the search results."
    )