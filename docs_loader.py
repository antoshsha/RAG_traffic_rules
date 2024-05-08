from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import MarkdownReader
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader
import re
import os
from llama_index.postprocessor.cohere_rerank import CohereRerank
load_dotenv()


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


doc_path_full = "data/traffic_rules"
doc_full = SimpleDirectoryReader(input_files=["data/full_traffic_rules.md"]).load_data()
doc_index = get_index(doc_full, "data/vectors/full_traffic_rules")

api_key = os.environ["COHERE_API_KEY"]
cohere_rerank = CohereRerank(api_key=api_key, top_n=4)
query_engine = doc_index.as_query_engine(
    similarity_top_k=10,
    temperature=0.1,
    node_postprocessors=[cohere_rerank],
)


doc_files = os.listdir("data/traffic_rules")
doc_files = sorted(doc_files)
doc_files = doc_files[:33]
doc_engines = {}
pattern = r"\d+_(.+)\.md"
doc_names = [re.match(pattern, file_name).group(1) for file_name in doc_files]

for idx, doc_file in enumerate(doc_files, start=1):
    doc_path = os.path.join("data/traffic_rules", doc_file)
    doc = MarkdownReader().load_data(file=doc_path)
    doc_index = get_index(doc, f"data/vectors/{doc_names[idx-1]}")
    doc_engine = doc_index.as_query_engine(
        similarity_top_k=10,
        temperature=0.1,
        node_postprocessors=[cohere_rerank],
    )
    doc_engines[f"doc_{idx}_engine"] = doc_engine



