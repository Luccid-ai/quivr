import os

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.node_parser import MarkdownElementNodeParser

from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

data_directory = "luccid-data/data/"
folder_name = "Documents/SerbiaGemini"
index_data = os.path.join(data_directory, folder_name, "index-data")

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]
llm = Gemini(model="models/gemini-1.5-pro", safety_settings=safety_settings)
embed_model = GeminiEmbedding(model_name="models/text-embedding-004")

Settings.llm = llm
Settings.embed_model = embed_model


class LlamaIndexBrain:
    """This is a first implementation of LlamaIndex recursive retriever RAG class. it is a KnowledgeBrainQA has the data is stored locally.
    It is going to call the Data Store internally to get the data.

    Args:
        KnowledgeBrainQA (_type_): A brain that store the knowledge internaly
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

    @classmethod
    def _load_data(cls, recursive: bool = False):
        reader = SimpleDirectoryReader(
            input_dir=os.path.join(data_directory, folder_name), recursive=recursive
        )
        docs = reader.load_data()

        return docs

    @classmethod
    def _parse_nodes(cls, docs):
        node_parser = MarkdownElementNodeParser(llm=llm)
        nodes = node_parser.get_nodes_from_documents(docs)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
        index = VectorStoreIndex(nodes=base_nodes + objects)
        index.set_index_id("vector_index")
        index.storage_context.persist(index_data)
        print(f"Ingested {len(nodes)} Nodes")


if __name__ == "__main__":
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=index_data
        )
    except ValueError as e:
        if (
            e
            == "No index in storage context, check if you specified the right persist_dir."
        ):
            docs = LlamaIndexBrain._load_data(recursive=True)
            LlamaIndexBrain._parse_nodes(docs=docs)
        else:
            print(e)
            # raise e
    except FileNotFoundError as e:
        print(f"### {e}")
        docs = LlamaIndexBrain._load_data(recursive=True)
        LlamaIndexBrain._parse_nodes(docs=docs)
