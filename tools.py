from llama_index.core.tools import QueryEngineTool, ToolMetadata
from docs_loader import *


tools = []

with open("data/combined_engine_description.txt", "r") as file:
    descriptions = file.readlines()

for idx, description in enumerate(descriptions, start=1):
    doc_engine = doc_engines[f"doc_{idx}_engine"]
    tool = QueryEngineTool(
        query_engine=doc_engine,
        metadata=ToolMetadata(
            name=f"doc_{idx}",
            description=description.strip(),
        ),
    )
    tools.append(tool)




