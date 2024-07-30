import os
import json
import time
from typing import AsyncIterable
from uuid import UUID

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from llama_index.core import (
    Settings,
    load_index_from_storage,
    StorageContext,
)

# from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate, PromptType

# from llama_index.core.ingestion import (
#     DocstoreStrategy,
#     IngestionCache,
#     IngestionPipeline,
# )
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# from llama_index.readers.google import GoogleDriveReader
# from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
# from llama_index.storage.docstore.redis import RedisDocumentStore
# from llama_index.vector_stores.redis import RedisVectorStore

from modules.brain.integrations.LlamaIndexSerbiaGemini.GeminiCustom import GeminiCustom
from modules.brain.knowledge_brain_qa import KnowledgeBrainQA
from modules.chat.dto.chats import ChatQuestion

from llama_index.embeddings.gemini import GeminiEmbedding

data_directory = "/data/"
folder_name = "Documents/Manufacturers/Velux-UK"
index_data = os.path.join(data_directory, folder_name, "index-data")
SYSTEM_INSTRUCTIONS = """
Primary Role: You are an experienced Serbian architect specializing in Serbian building codes, regulations, and standards.
Instructions for Responses:
    - Clarity and Conciseness: Responses should be clear, concise, and professional. Avoid technical jargon unless it is necessary for understanding.
    - Reference to Regulations: Always cite specific regulations, articles, or standards from Serbian building codes.
    - Relevance: Respond only to questions directly related to building regulations. If a question is not related to architecture or building regulations, kindly inform the user that the question is outside your area of expertise.
    - Precision: Adhere exclusively to Serbian building codes, regulations, and standards as primary sources.
    - Timeliness: Use the most recent regulations in case of conflict or uncertainty. If some regulation might be updated or will soon be updated, mention that as well.
    - Neutrality: Avoid personal opinions or advice that is not in accordance with the regulations.
    - Adaptability: Tailor responses to the user's level of understanding. If the user is not an expert, explain regulations in simple terms. If information is insufficient, specify what additional information is needed.
    - Use of Ekavica: Responses should be written exclusively in Ekavica, using Serbian Latin script. Avoid using Ijekavica.

    Goal: To assist architects and professionals in complying with Serbian building regulations.

Few-shot examples:
    <examples>
        Query: Koja je najmanja dimenzija parking mesta za automobile?
        Response: Najmanja dimenzija parking mesta za parkiranje je 230/480 cm, parking mesta za podužno parkiranje je 200/550 cm, a garažnog boksa 270/550 cm.

        Query: Kako se ocenjuje stepen otpornosti objekta prema požaru?
        Response: Stepen otpornosti objekta prema požaru je ocena ponašanja objekta na dejstvo požara i izražava se ocenama od I do V, odnosno: neznatan (I), mali (II), srednji (III), veći (IV), veliki (V).

        Query: Kada je potrebno da član komisije za tehnički pregled bude inženjer protivpožarne zaštite?
        Response: Član komisije za tehnički pregled treba da bude inženjer protivpožarne zaštite sa odgovarajućom licencom kada je predmet tehničkog pregleda objekat za koji su utvrđene posebne mere zaštite od požara.

        Query: Koje su minimalne dimenzije evakuacionih vrata?
        Response: Minimalna širina svetlog otvora vrata stanova, kancelarija i sl. u kojima boravi do deset lica iznosi 0,90 m. Minimalna širina svetlog otvora vrata prostorija u kojima boravi više od deset lica, a manje od pedeset lica, iznosi 1 m. Visina vrata na svim evakuacionim putevima je najmanje 2 m, a u javnim objektima najmanje 2,10 m. Za prostorije u kojima boravi više od 50 a manje od 100 lica primenjuju se dvokrilna vrata ili dvoje jednokrilnih vrata na adekvatnom rastojanju. Za prostorije u kojima boravi više od 100 lica primenjuje se više dvokrilnih i/ili jednokrilnih vrata.

        Query: Za koje objekte se izrađuje izveštaj o zatečenom stanju?
        Response: Izveštaj o zatečenom stanju objekta se izrađuje za: objekte kategorije A, klase 111011 i 112111, objekte čija je bruto razvijena građevinska površina objekta (u daljem tekstu: BRGP) veća od 400 m2, objekte javne namene, inženjerske objekte.

        Query: Kako se označavaju projekti u tehničkoj domunetaciji?
        Response: Projekti su u tehničkoj dokumentaciji označeni rednim brojem i obavezno složeni u sveske, prema sledećim oblastima i redosledu: broj "1": arhitektura, broj "2": građevinski projekti, broj "3": hidrotehničke instalacije, broj "4": elektroenergetske instalacije, broj "5": telekomunikacione i signalne instalacije, broj "6": mašinske instalacije, broj "7": tehnologija, broj "8": saobraćaj i saobraćajna signalizacija, broj "9": spoljno uređenje sa sinhron-planom instalacija i priključaka, pejzažna arhitektura i hortikultura, broj "10": pripremni radovi (rušenje, zemljani radovi, obezbeđenje temeljne jame). Projekat priključka na javnu komunalnu infrastrukturu je deo projekta odgovarajuće oblasti, odnosno vrste instalacija. Svaki projekat određene oblasti se može deliti na više svezaka koje dobijaju odgovarajuće oznake u zavisnosti odsadržaja projekta (na primer: 2/1 konstrukcija, 2/2 saobraćajnice i dr., 3/1 vodovod, 3/2 kanalizacija i dr., 6/1 grejanje, 6/2ventilacija i klimatizacija itd.).

    </examples>
"""

storage_context = None
index = None
reranker = None

if os.path.exists(index_data):
    try:
        if not storage_context:
            print("####### Starting loading storage context... #######")
            start_time = time.time()  # Record the start time

            storage_context = StorageContext.from_defaults(persist_dir=index_data)

            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(
                f"####### Finishing loading storage context... in {elapsed_time:.2f} seconds #######"
            )
        if not index:
            print("####### Starting loading index from storage... #######")
            start_time = time.time()  # Record the start time

            index = load_index_from_storage(
                storage_context=storage_context, index_id="vector_index"
            )

            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(
                f"####### Finishing loading index from storage... in {elapsed_time:.2f} seconds #######"
            )
        if not reranker:
            print("####### Starting loading reranker... #######")
            start_time = time.time()  # Record the start time

            reranker = FlagEmbeddingReranker(
                top_n=7, model="BAAI/bge-reranker-large", use_fp16=True
            )

            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(
                f"####### Finishing loading reranker... in {elapsed_time:.2f} seconds #######"
            )
    except ValueError as e:
        print(e)
    except FileNotFoundError as e:
        print(f"### {e}")
else:
    print("### No index found...")

# embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# llm = OpenAI(model="gpt-4o")

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
llm = GeminiCustom(model="models/gemini-1.5-pro", safety_settings=safety_settings, system_instructions=SYSTEM_INSTRUCTIONS)
embed_model = GeminiEmbedding(model_name="models/text-embedding-004")

Settings.llm = llm
Settings.embed_model = embed_model


class LlamaIndexVeluxUK(KnowledgeBrainQA):
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
        self._storage_context = storage_context
        self._index = index
        self._reranker = reranker

    def _get_engine(self):
        if not self._index:
            print("### No index found...")
            return None

        VELUX_TEXT_QA_PROMPT_TMPL = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Only use information and product details provided in the context."
            "You are an experienced architect specializing in Velux products (building and construction products, TODO(pg)...)."
            "You will answer in Professional architectural and building and construction products Language."
            "Keep your answers short and always deliver only what was asked."
            "Be as descriptive as possible. Always make sure to provide 100% correct information."
            "When responding, avoid giving personal opinions or advice that goes beyond the scope of regulations."
            "In cases of conflicting information, use the most recent product by the date of being published."
            "Your responses should be clear, concise, and tailored to the level of understanding of the user, ensuring they receive the most relevant and accurate information."
            "Your goal is to help architects with building regulations so they don't get rejected by the building inspectorate."
            "Always answer in the language you were spoken to unless the user speaks in serbian or croatian, then always answer in latin serbian."
            "Query: {query_str}\n"
        )
        VELUX_TEXT_QA_PROMPT = PromptTemplate(
            VELUX_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
        )
        VELUX_SYSTEM_PROMPT_TMPL = VELUX_TEXT_QA_PROMPT_TMPL
        VELUX_SYSTEM_PROMPT = PromptTemplate(
            VELUX_SYSTEM_PROMPT_TMPL, prompt_type=PromptType.CUSTOM
        )

        return self._index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            similarity_top_k=15,
            node_postprocessors=[self._reranker],
            text_qa_template=VELUX_TEXT_QA_PROMPT,
            system_prompt=VELUX_SYSTEM_PROMPT,
            stream=True,
            verbose=True,
        )

    def _format_chat_history(self, chat_history):
        return [
            ChatMessage(
                role=(
                    MessageRole.USER
                    if isinstance(message, HumanMessage)
                    else (
                        MessageRole.ASSISTANT
                        if isinstance(message, AIMessage)
                        else (
                            MessageRole.SYSTEM
                            if isinstance(message, SystemMessage)
                            else MessageRole.MODEL
                        )
                    )
                ),
                content=message.content,
            )
            for message in chat_history
        ]

    async def generate_stream(
        self, chat_id: UUID, question: ChatQuestion, save_answer: bool = True
    ) -> AsyncIterable:
        print(f"####### Calling generate_stream with question: {question} #######")
        chat_engine = self._get_engine()
        if not chat_engine:
            raise ValueError("No chat engine found")
        transformed_history, streamed_chat_history = (
            self.initialize_streamed_chat_history(chat_id, question)
        )
        print(f"####### transformed_history: {transformed_history} #######")
        llama_index_transformed_history = self._format_chat_history(transformed_history)

        response_tokens = []
        # response = await chat_engine.astream_chat(
        #     message=question.question,
        #     chat_history=llama_index_transformed_history,
        # )
        response = chat_engine.stream_chat(
            message=question.question,
            chat_history=llama_index_transformed_history,
        )
        for chunk in response.response_gen:
            print(chunk)
            response_tokens.append(chunk)
            streamed_chat_history.assistant = chunk
            yield f"data: {json.dumps(streamed_chat_history.dict())}"
        # response = await chat_engine.aquery(
        #     question.question,
        # )
        # streamed_chat_history.assistant = str(response)
        # yield f"data: {json.dumps(streamed_chat_history.dict())}"

        # self.save_answer(question, str(response), streamed_chat_history, save_answer)
        self.save_answer(question, response_tokens, streamed_chat_history, save_answer)
