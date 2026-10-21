from llm import llm
from prompts import responder_prompt, revisor_prompt
from schemas import AnswerQuestion, ReviseAnswer

responder_chain = responder_prompt | llm.with_structured_output(AnswerQuestion)
revisor_chain = revisor_prompt | llm.with_structured_output(ReviseAnswer)