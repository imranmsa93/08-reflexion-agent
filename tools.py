from dotenv import load_dotenv

load_dotenv()
from langchain_tavily import TavilySearch

tavily = TavilySearch(max_results = 2)

def perform_search(queries:list[str]) -> str:
    blocks = []
    for query in queries[:3]:
        query = query.replace("-"," ")
        reponse = tavily.invoke({"query":query})
        for r in reponse["results"]:
            blocks.append(f'Source:{r.get("url")}\n{r.get("content","")}')
    return "\n\n".join(blocks)