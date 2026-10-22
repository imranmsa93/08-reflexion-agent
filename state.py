from typing import List, TypedDict


class ReflexionState(TypedDict, total=False):
    question: str              # what the user asked
    answer: str                # the current best answer
    missing: str               # critique: what is missing
    superfluous: str           # critique: what is unnecessary
    search_queries: List[str]  # queries the model wants to run
    search_results: str        # text returned by the search
    references: List[str]      # source URLs for the final answer
    rounds: int                # how many revisions have happened