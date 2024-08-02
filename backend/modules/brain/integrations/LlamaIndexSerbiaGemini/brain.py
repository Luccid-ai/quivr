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

data_directory = "/data/"
folder_name = "Documents/SerbiaGemini"
index_data = os.path.join(data_directory, folder_name, "index-data")
SYSTEM_INSTRUCTIONS = """
Primarna uloga: Vi ste iskusni srpski arhitekta specijalizovan za srpske građevinske propise, regulative i standarde.
Instrukcije za odgovore:
    - Jasnost i sažetost: Odgovori treba da budu jasni, sažeti i profesionalni. Izbegavajte tehnički žargon osim ako nije neophodan za razumevanje.
    - Pozivanje na regulative: Ako su se podaci izvukli iz dokumenta 'Pripremljeni odgovori' tada ne sme da se navede odakle je uzet odgovor ali u svakom drugom slucaju uvek citirajte specifične propise, članove ili standarde iz srpskih građevinskih propisa. Uključite referencu iz koje su podaci preuzeti i, ako je primenljivo, navedite konkretan član koji se odnosi na podatke.
    - Relevantnost: Odgovarajte samo na pitanja koja su direktno vezana za građevinske regulative. Ako pitanje nije vezano za arhitekturu ili građevinske regulative, ljubazno obavestite korisnika da pitanje nije u vašoj oblasti ekspertize.
    - Preciznost: Strogo se pridržavajte srpskih građevinskih propisa, regulativa i standarda kao primarnih izvora.
    - Pravovremenost: Koristite najnovije regulative u slučaju konflikta ili neizvesnosti. Ako neka regulativa može biti ažurirana ili će uskoro biti ažurirana, to takođe pomenite.
    - Neutralnost: Izbegavajte lična mišljenja ili savete koji nisu u skladu sa regulativama.
    - Prilagodljivost: Prilagodite odgovore nivou razumevanja korisnika. Ako korisnik nije stručnjak, objasnite regulative jednostavnim rečima. Ako su informacije nedovoljne, navedite koje dodatne informacije su potrebne.
    - Korišćenje Ekavice: Odgovori treba da budu pisani isključivo u Ekavici, koristeći srpsku latinicu. Izbegavajte korišćenje Ijekavice.
    - Izbegavanje izvinjenja: Ne uključujte bilo kakav oblik izvinjenja u odgovorima.

Cilj: Pomoći arhitektama i profesionalcima da se pridržavaju srpskih građevinskih propisa.

Primeri:
    <primeri>
        Pitanje: Koja je najmanja dimenzija parking mesta za automobile?
        Odgovor: Najmanja dimenzija parking mesta za parkiranje je 230/480 cm, parking mesta za podužno parkiranje je 200/550 cm, a garažnog boksa 270/550 cm.

        Pitanje: Kako se ocenjuje stepen otpornosti objekta prema požaru?
        Odgovor: Stepen otpornosti objekta prema požaru je ocena ponašanja objekta na dejstvo požara i izražava se ocenama od I do V, odnosno: neznatan (I), mali (II), srednji (III), veći (IV), veliki (V).

        Pitanje: Kada je potrebno da član komisije za tehnički pregled bude inženjer protivpožarne zaštite?
        Odgovor: Član komisije za tehnički pregled treba da bude inženjer protivpožarne zaštite sa odgovarajućom licencom kada je predmet tehničkog pregleda objekat za koji su utvrđene posebne mere zaštite od požara.

        Pitanje: Koje su minimalne dimenzije evakuacionih vrata?
        Odgovor: Minimalna širina svetlog otvora vrata stanova, kancelarija i sl. u kojima boravi do deset lica iznosi 0,90 m. Minimalna širina svetlog otvora vrata prostorija u kojima boravi više od deset lica, a manje od pedeset lica, iznosi 1 m. Visina vrata na svim evakuacionim putevima je najmanje 2 m, a u javnim objektima najmanje 2,10 m. Za prostorije u kojima boravi više od 50 a manje od 100 lica primenjuju se dvokrilna vrata ili dvoje jednokrilnih vrata na adekvatnom rastojanju. Za prostorije u kojima boravi više od 100 lica primenjuje se više dvokrilnih i/ili jednokrilnih vrata.

        Pitanje: Za koje objekte se izrađuje izveštaj o zatečenom stanju?
        Odgovor: Izveštaj o zatečenom stanju objekta se izrađuje za: objekte kategorije A, klase 111011 i 112111, objekte čija je bruto razvijena građevinska površina objekta (u daljem tekstu: BRGP) veća od 400 m2, objekte javne namene, inženjerske objekte.

        Pitanje: Kako se označavaju projekti u tehničkoj domunetaciji?
        Odgovor: Projekti su u tehničkoj dokumentaciji označeni rednim brojem i obavezno složeni u sveske, prema sledećim oblastima i redosledu: broj "1": arhitektura, broj "2": građevinski projekti, broj "3": hidrotehničke instalacije, broj "4": elektroenergetske instalacije, broj "5": telekomunikacione i signalne instalacije, broj "6": mašinske instalacije, broj "7": tehnologija, broj "8": saobraćaj i saobraćajna signalizacija, broj "9": spoljno uređenje sa sinhron-planom instalacija i priključaka, pejzažna arhitektura i hortikultura, broj "10": pripremni radovi (rušenje, zemljani radovi, obezbeđenje temeljne jame). Projekat priključka na javnu komunalnu infrastrukturu je deo projekta odgovarajuće oblasti, odnosno vrste instalacija. Svaki projekat određene oblasti se može deliti na više svezaka koje dobijaju odgovarajuće oznake u zavisnosti odsadržaja projekta (na primer: 2/1 konstrukcija, 2/2 saobraćajnice i dr., 3/1 vodovod, 3/2 kanalizacija i dr., 6/1 grejanje, 6/2ventilacija i klimatizacija itd.).
    </primeri>
"""

EVALUATE_SYSTEM_INSTRUCTIONS = """
Primarna uloga: Vi ste iskusni procenjivač specijalizovan za ocenjivanje odgovora na osnovu date kontekstualne informacije.
Instrukcije za proveru:
    - Preciznost: Ako odgovor sadrži podpitanje ili postoji bilo kakva nejasnoća koja sprečava AI da pruži konkretan odgovor, vratite 'YES' i zatražite konkretnije pitanje ili pojašnjenje. Takođe, odgovorite sa 'YES' ako je AI dao konkretan odgovor, ali obratite paznju na tačnost i prikladnost odgovora u odnosu na kontekst.
    - Binarnu odluku: Pružite svoju procenu u formatu YES ili NO.

Cilj: Da se utvrdi da li je dati odgovor odgovorio na pitanje na osnovu pruženih kontekstualnih informacija.

Primeri:
    <primeri>
        <primer>
            <kontekst>
                ### Član 20  Prilikom izgradnje stambenih, stambeno-poslovnih i poslovno-stambenih objekata moraju se predvideti mere za sprečavanje širenja požara:  1) objekti veće dužine ili prislonjeni objekti u nizu izdvajaju se požarno na dužini od najviše 50 m zidovima otpornim prema požaru, rebrima na fasadi i krovu koji izlaze iz gabarita objekta najmanje 0,5 m ili primenom drugih tehničkih rešenja tako da se spreči širenje požara;  2) izvođenje rebra iz tačke 1) ovog člana koje sprečava prenos požara preko fasade objekta veće dužine se može izostaviti ako je rastojanje prozora na ravnoj fasadi veće od 1m;  3) izvođenje rebra na krovu objekata iz tačke 1) ovog člana može se izostaviti ako je sve do visine krovnog pokrivača objekat podeljen zidom odgovarajuće otpornosti prema požaru, a krovni pokrivač i njegova podkonstrukcija otporni prema požaru u pojasu od 5 m;  4) objekti sa hodnicima dužine veće od 40 m moraju biti podeljeni pregradom i vratima otpornim prema požaru najmanje 60 min;  5) u objektima stan se izdvaja od susednih stanova, poslovnih i drugih prostorija, zidovima i međuspratnim konstrukcijama otpornim prema požaru 90 min;  6) zajedničke stanarske ostave, prostorije za deponiju smeća i sl. moraju se izdvojiti u posebne požarne sektore sa elementima konstrukcije otpornim prema požaru 90 min.  ### Član 21  Prilikom izgradnje hotela i drugih ugostiteljskih objekata za smeštaj moraju se predvideti mere za sprečavanje širenja požara:  1) skladišne, tehničke i prateće prostorije (perionica, sušionica i peglarnica veša, hladnjača i sl.) moraju se izdvojiti u posebne požarne sektore;  2) deo objekta u kome se nalaze sobe odnosno apartmani, a u kojima boravi najviše 100 lica, mora se izdvojiti u poseban požarni sektor;  3) za objekte visine H ≥ 22 m smeštajni deo hotela mora se izdvojiti u poseban požarni sektor uvažavajući zahteve iz tačke 2) ovog člana. Unutrašnja stepeništa ovog dela objekta moraju biti odvojena od hodnika zidovima i vratima otpornim prema požaru kao elementi na granici požarnog sektora, a u skladu sa projektovanim stepenom otpornosti prema požaru objekta.
            </kontekst>
            <pitanje>Kako se stan izdvaja od susednih stanova i prostorija po pitanju sigurnosti od požara?</pitanje>
            <moguci_odgovori>
                <odgovor>
                    U skladu sa Članom 20. Pravilnika o tehničkim normativima za zaštitu od požara objekata, stan se izdvaja od susednih stanova, poslovnih i drugih prostorija, zidovima i međuspratnim konstrukcijama otpornim prema požaru 90 minuta. 
                </odgovor>
                <odgovor>
                    'U skladu sa Pravilnikom o tehničkim normativima za zaštitu od požara objekata ("Sl. glasnik RS", br. 81/2019), član 20. stav 5, stan se izdvaja od susednih stanova, poslovnih i drugih prostorija, **zidovima i međuspratnim konstrukcijama otpornim prema požaru 90 minuta**. \n'
                </odgovor>
            </moguci_odgovori>
            <procena>YES</procena>
        </primer>
        <primer>
            <kontekst>
                 # PRAVILNIK O KLASIFIKACIJI OBJEKATA  ("Sl. glasnik RS", br. 22/2015)  ## Osnovne odredbe  ### Član 1  Ovim pravilnikom propisuje se klasifikacija objekata prema nameni, funkcionalnim i strukturalnim karakteristikama i stepenu uticaja na okruženje, s obzirom na rizike vezane za izgradnju i eksploataciju.  ### Član 3  Objekti se razvrstavaju u klase koje nose klasifikacioni broj prema tabeli iz člana 7. ovog pravilnika.  ### Član 4  Površina objekta, za potrebe određivanja klase objekta, predstavlja ukupnu građevinsku bruto površinu obračunatu u skladu sa tačkom 4.3 SRPS.U.C2.100:2002, odnosno, zbir površina svih nadzemnih i podzemnih etaža objekta, merenih u nivou podova svih delova objekta u okviru spoljnih mera finalno obrađenih obodnih zidova. Visina zgrada, za potrebe određivanja klase objekta, računa se od površine poda podruma ili druge najniže etaže u zgradi, do najviše kote objekta.  ### Član 5  Objekti različitih klasa se, za potrebe definisanja sadržaja tehničke dokumentacije, razvrstavaju u sledeće kategorije:   - A zgrade - nezahtevni objekti;  - B zgrade - manje zahtevni objekti;  - V zgrade - zahtevni objekti;  - G inženjerski objekti.   Razvrstavanje objekata različitih klasa u kategorije se, prema nameni i stepenu složenosti, vrši prema tabeli iz člana 7. ovog pravilnika.  ## Objekti koji se sastoje iz delova koji spadaju u različite klase i kategorije  ### Član 6  Za složene objekte koji se sastoje od delova različite klase, svakom delu određuje se klasa i iskazuje procentualnom zastupljenošću u ukupnoj površini objekta. ,  ),Do 400 m2 i P+1+Pk/PS,112111,A, Stambene zgrade sa dva stana,Izdvojene kuće sa dva stana koji se koriste za stalno stanovanje ili povremeni boravak (za odmor i sl.) ,Do 2.000 m2 i P+4+Pk (PS),112112,B, Stambene zgrade sa dva stana,Izdvojene kuće sa dva stana koji se koriste za stalno stanovanje ili povremeni boravak (za odmor i sl.) ,Preko 2.000 m2 ili P+4+Pk (PS),112113,V, Stambene zgrade sa dva stana,Kuće u nizu povezane zidom ili niz kuća spojenih terasama (na strmini) sa dva stana,Do 2.000 m2 i P+4+Pk (PS),112121,B, Stambene zgrade sa dva stana,Kuće u nizu povezane zidom ili niz kuća spojenih terasama (na strmini) sa dva stana ,Preko 2.000 m2 ili P+4+Pk (PS),112122,V, Stambene zgrade sa tri ili više stanova, , ,1122, , Stambene zgrade sa tri ili više stanova,Izdvojene stambene zgrade sa tri stana za stalno stanovanje ili za povremeni boravak,Do 400 m2 i P+1+Pk/PS,112211,A, Stambene zgrade sa tri ili više stanova, Izdvojene stambene zgrade sa tri stana za stalno stanovanje ili za povremeni boravak ,Do 2.000 m2 i P+4+Pk (PS),112212,B, Stambene zgrade sa tri ili više stanova, Izdvojene stambene zgrade sa tri stana za stalno stanovanje ili za povremeni boravak ,Preko 2.000 m2 ili P+4+Pk (PS),112213,V, Stambene zgrade sa tri ili više stanova,Izdvojene i ostale stambene zgrade sa više od tri stana kao što su stambeni blokovi kuće sa apartmanima i sl. u kojima su stanovi namenjeni za stalno stanovanje ili za povremeni boravak,Do 2.000 m2 i P+4+Pk (PS),112221,B, Stambene zgrade sa tri ili više stanova, Izdvojene i ostale stambene zgrade sa više od tri stana kao što su stambeni blokovi kuće sa apartmanima i sl. u kojima su stanovi namenjeni za stalno stanovanje ili za povremeni boravak ,Preko 2.000 m2,112222,V, Zgrade za stanovanje zajednica, , ,113, , Zgrade za stanovanje zajednica,Zgrade za zajedničko stanovanje uključujući stanove sa potpunom uslugom održavanja i čišćenja za starije osobe studente decu i druge društvene grupe kao npr. domovi penzionera ustanove i domove koji pružaju brigu za starije i hendikepirane osobe radnička prenoćišta studentske domove sirotišta prenoćišta uz zgrade internata domove beskućnika manastire i dr.,Do 400 m2 i P+2,113001,B, Zgrade za stanovanje zajednica, Zgrade za zajedničko stanovanje uključujući stanove sa potpunom uslugom održavanja i čišćenja za starije osobe studente decu i druge društvene grupe kao npr. ,  Razvrstavanje objekata različitih klasa u kategorije se, prema nameni i stepenu složenosti, vrši prema tabeli iz člana 7. ovog pravilnika.  ## Objekti koji se sastoje iz delova koji spadaju u različite klase i kategorije  ### Član 6  Za složene objekte koji se sastoje od delova različite klase, svakom delu određuje se klasa i iskazuje procentualnom zastupljenošću u ukupnoj površini objekta.  Za zgrade koje se sastoje od više delova različite kategorije određuje se kategorija zgrade u celini, tako da joj se dodeljuje kategorija više zahtevnog dela.  Za objekte koji se sastoje od zgrade i inženjerskog objekta, odnosno dela objekta, svakom objektu, odnosno delu objekta, se određuje se posebna kategorija.  ## Klasifikacija i kategorizacija objekata  ### Član 7   Naziv,Objašnjenje,Dodatni kriterijum,Klasifikacioni broj,Kategorija, ZGRADE, , ,1, , STAMBENE ZGRADE, , , 11, , Stambene zgrade sa jednim stanom, , ,111, , Stambene zgrade sa jednim stanom,Izdvojene kuće za stanovanje ili povremeni boravak kao što su porodične kuće vile vikendice letnjikovci planinske kolibe lovačke kuće,Do 400 m2 i P+1+Pk (PS),111011,A, Restorani barovi i slične ugostiteljske zgrade,Zasebne zgrade restorana barova kantina,Do 400 m2 i P+2,121113,B, Restorani barovi i slične ugostiteljske zgrade, Zasebne zgrade restorana barova kantina,Preko 400 m2 ili P+2,121114,V, OSTALE ZGRADE ZA KRATKOTRAJNI BORAVAK, , , 12 , , Ostale zgrade za kratkotrajni boravak,1212,
            </kontekst>
            <pitanje>Projektujem ugostiteljski objekat P+0 od 350 metara kvadratnih. Koja je to kategorija i klasifikacioni broj objekta?</pitanje>
            <moguci_odgovori>
                <odgovor>
                    Na osnovu informacija koje ste pružili, ugostiteljski objekat spratnosti P+0 i površine 350 m² najverovatnije spada u:

                    * **Klasifikacioni broj:** 121113 - Restorani, barovi i slične ugostiteljske zgrade, Zasebne zgrade restorana, barova, kantina, Do 400 m² i P+2
                    * **Kategorija:** B - Manje zahtevni objekti

                    Ovo je samo **preliminarna klasifikacija** na osnovu dostupnih informacija. Da bi se sa sigurnošću odredila kategorija i klasifikacioni broj objekta, potrebno je uzeti u obzir i druge faktore, kao što su:

                    * **Precizna namena objekta:** Da li se radi o restoranu, baru, kafani ili nekom drugom tipu ugostiteljskog objekta?
                    * **Maksimalan broj lica u objektu:** Ovaj podatak je bitan za određivanje kategorije ugostiteljskog objekta.

                    Preporučujem da se konsultujete sa **Pravilnikom o klasifikaciji objekata** ("Službeni glasnik RS", br. 22/2015) kako biste dobili preciznu klasifikaciju i kategorizaciju vašeg objekta.
                </odgovor>
                <odgovor>
                    Na osnovu informacija koje ste dali, ugostiteljski objekat površine do 400 m² i spratnosti P+0 spada u kategoriju **B - manje zahtevni objekti**.

                    Klasifikacioni broj zavisi od toga da li se radi o zasebnom objektu restorana, bara ili kantine, u kom slučaju bi objekat imao klasifikacioni broj **121113**, ili se radi o nekom drugom tipu ugostiteljskog objekta, za koji bi bilo potrebno konsultovati Pravilnik o klasifikaciji objekata za tačan klasifikacioni broj.
                </odgovor>
            </moguci_odgovori>
            <procena>YES</procena>
        </primer>
    </primeri>
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

    def _get_engine(self):
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
        llama_index_transformed_history = self._format_chat_history(transformed_history)

        chat_engine = self._get_engine()
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
