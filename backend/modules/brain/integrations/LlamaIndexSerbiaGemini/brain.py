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

from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from llama_index.core.evaluation import RelevancyEvaluator
# from llama_index.readers.google import GoogleDriveReader
# from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
# from llama_index.storage.docstore.redis import RedisDocumentStore
# from llama_index.vector_stores.redis import RedisVectorStore

from modules.brain.integrations.LlamaIndexSerbiaGemini.gemini_custom import GeminiCustom
from modules.brain.integrations.LlamaIndexSerbiaGemini.retry_context_chat_engine import CustomChatEngine
from llama_index.core.chat_engine.context import ContextChatEngine
from modules.brain.knowledge_brain_qa import KnowledgeBrainQA
from modules.chat.dto.chats import ChatQuestion
from arize_otel import register_otel, Endpoints
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

data_directory = "luccid-data/data/"
folder_name = "Documents/SerbiaGemini"
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

EVALUATE_SYSTEM_INSTRUCTIONS = """
Primary Role: You are an experienced assessor specializing in evaluating responses based on given contextual information.
Instructions for Responses:
    -Clarity and Conciseness: Responses should be clear, concise, and to the point. Avoid unnecessary elaboration.
    -Consistency: Check if the response is aligned with the context provided.
    -Relevance: Ensure the response directly answers the question based on the provided context.
    -Binary Decision: Provide your assessment in a YES or NO format.
    -Precision: Focus on the accuracy and appropriateness of the response in relation to the context.
    -Neutrality: Maintain objectivity and avoid adding personal opinions or external information not included in the provided context.
    -Timeliness: Ensure that the most current information is considered if relevant.
    -Avoid Apologizing: Do not include any form of apology in the responses.

Goal: To determine if the given response is consistent with the provided contextual information.

Few-shot examples:
    <examples>
        Context: The sky is clear and blue.
        Query: What color is the sky?
        Response: The sky is blue.
        Assessment: YES

        ruby
        Copy code
        **Context:** The user requested information about the weather.
        **Query:** What is the temperature today?
        **Response:** The temperature is 25°C.
        **Assessment:** YES

        **Context:** The capital of France is Paris.
        **Query:** What is the capital of France?
        **Response:** The capital of France is London.
        **Assessment:** NO

        **Context:** The project deadline is next Friday.
        **Query:** When is the project deadline?
        **Response:** The project deadline is next Friday.
        **Assessment:** YES

        **Context:** The instructions state to use only blue ink.
        **Query:** What color ink should be used?
        **Response:** You should use red ink.
        **Assessment:** NO
    </examples>
"""

storage_context = None
index = None
reranker = None

# Setup OTEL via our convenience function
register_otel(
    endpoints = Endpoints.ARIZE,
    space_key = "224c3ca", # in app space settings page
    api_key = "b578ce267ae08994b8b", # in app space settings page
    model_id = "SerbianBrain", # name this to whatever you would like
)

# Finish automatic instrumentation
LlamaIndexInstrumentor().instrument()

if os.path.exists(index_data):
    try:
        if not storage_context:
            print("####### Starting loading storage context... #######")
            start_time = time.time()  # Record the start time

            storage_context = StorageContext.from_defaults(
                persist_dir=index_data
            )

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


class LlamaIndexSerbiaGemini(KnowledgeBrainQA):
    """This is a first implementation of LlamaIndex recursive retriever RAG class. it is a KnowledgeBrainQA has the data is stored locally.
    It is going to call the Data Store internally to get the data.

    Args:
        KnowledgeBrainQA (_type_): A brain that store the knowledge internaly
    """

    def __init__(
        self,
        eval_llm: GeminiCustom = GeminiCustom(model="models/gemini-1.5-pro", safety_settings=safety_settings, system_instructions=EVALUATE_SYSTEM_INSTRUCTIONS),
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self._storage_context = storage_context
        self._index = index
        self._reranker = reranker
        self._eval_llm = eval_llm

    def _get_engine(self, chat__history):
        if not self._index:
            print("### No index found...")
            return None

        DEFAULT_TEXT_QA_PROMPT_TMPL = (
            "Contextual Information:\n"
                    "{context_str} - This can include plot size, building type, location, soil conditions, or existing planning documents.\n"
            "Query:\n"
            "{query_str}\n\n"
        )
        DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
            DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
        )

        EVALUATOR_PROMPT_TMPL = (
            "Query and Response:"
            "{query_str}"
            "Context:"
            "{context_str}"
            "Answer:"
        )

        query_response_evaluator = RelevancyEvaluator(
            llm=self._eval_llm,
            eval_template=EVALUATOR_PROMPT_TMPL
        )

        context_chat_engine = self._index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            similarity_top_k=20,
            node_postprocessors=[self._reranker],
            text_qa_template=DEFAULT_TEXT_QA_PROMPT,
            stream=True,
            verbose=True,
        )

        if isinstance(context_chat_engine, ContextChatEngine):
            return CustomChatEngine(context_chat_engine=context_chat_engine, evaluator=query_response_evaluator, max_retries=1)
        
        return None

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
        transformed_history, streamed_chat_history = (
            self.initialize_streamed_chat_history(chat_id, question)
        )
        print(f"####### transformed_history: {transformed_history} #######")
        llama_index_transformed_history = self._format_chat_history(transformed_history[-5:])

        chat_engine = self._get_engine(llama_index_transformed_history)
        if not chat_engine:
            raise ValueError("No chat engine found")

        response_tokens = []

        response = await chat_engine.achat(
            message=question.question,
            chat_history=llama_index_transformed_history,
        )

        chunk_size = 3
        for i in range(0, len(response.response), chunk_size):
            chunk = response.response[i:i+chunk_size]
            response_tokens.append(chunk)
            streamed_chat_history.assistant = chunk
            yield f"data: {json.dumps(streamed_chat_history.dict())}"

        if len(response_tokens) > 0:
            self.save_answer(question, response_tokens, streamed_chat_history, save_answer)
