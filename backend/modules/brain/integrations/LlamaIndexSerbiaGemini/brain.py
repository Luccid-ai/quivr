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
        Context: # Član 3\r\n\r\nObjekti se razvrstavaju u klase koje nose klasifikacioni broj prema tabeli iz člana 7. ovog pravilnika.\r\n\r\n### Član 5\r\n\r\nObjekti različitih klasa se, za potrebe definisanja sadržaja tehničke dokumentacije, razvrstavaju u sledeće kategorije: \r\n\r\n- A zgrade - nezahtevni objekti; \r\n- B zgrade - manje zahtevni objekti; \r\n- V zgrade - zahtevni objekti; \r\n- G inženjerski objekti. \r\n\r\nRazvrstavanje objekata različitih klasa u kategorije se, prema nameni i stepenu složenosti, vrši prema tabeli iz člana 7. ovog pravilnika.\r\n\r\n## Objekti koji se sastoje iz delova koji spadaju u različite klase i kategorije\r\n\r\n### Član 6\r\n\r\nZa složene objekte koji se sastoje od delova različite klase, svakom delu određuje se klasa i iskazuje procentualnom zastupljenošću u ukupnoj površini objekta.domovi penzionera ustanove i domove koji pružaju brigu za starije i hendikepirane osobe radnička prenoćišta studentske domove sirotišta prenoćišta uz zgrade internata domove beskućnika manastire i dr.,Do 400 m2 i P+2,113001,B,\r\nZgrade za stanovanje zajednica, Zgrade za zajedničko stanovanje uključujući stanove sa potpunom uslugom održavanja i čišćenja za starije osobe studente decu i druge društvene grupe kao npr. domovi penzionera ustanove i domove koji pružaju brigu za starije i hendikepirane osobe radnička prenoćišta studentske domove sirotišta prenoćišta uz zgrade internata domove beskućnika manastire i dr.Preko 400 m2 ili P+2,113002,V,\r\nNESTAMBENE ZGRADE, , , 12 , ,\r\nHoteli i slične zgrade, , ,121, ,\r\nHoteli, , ,1211, ,\r\nHoteli i moteli,Hoteli moteli gostionice sa sobama pansioni i slične zgrade za noćenje gostiju s restoranom ili bez njega,Do 400 m2 i P+2,121111,B,\r\nHoteli i moteli,Hoteli moteli gostionice sa sobama pansioni i slične zgrade za noćenje gostiju s restoranom ili bez njega ,Preko 400 m2 ili P+2,121112,V,\r\nRestorani barovi i slične ugostiteljske zgrade,Zasebne zgrade restorana barova kantina,Do 400 m2 i P+2,121113,B,\r\nRestorani barovi i slične ugostiteljske zgrade, Zasebne zgrade restorana barova kantina,Preko 400 m2 ili P+2,121114,V,\r\nOSTALE ZGRADE ZA KRATKOTRAJNI BORAVAK, , , 12 , ,\r\nOstale zgrade za kratkotrajni boravak,1212,\r\nOstale zgrade za kratkotrajni boravak,Prenoćišta za omladinu (hostele) planinarski domovi dečiji i porodični kampovi bungalovi za odmor odmarališta druge zgrade za odmor i noćenje izletnika koje nisu drugde razvrstane,Do 400 m2 i P+2,121201 ,B,\r\n
        Query: Projektujem ugostiteljski objekat P+0 od 350 metara kvadratnih. Koja je to kategorija i klasifikacioni broj objekta?
        Response: Na osnovu informacija koje ste dali, ugostiteljski objekat površine do 400 m² i spratnosti P+0 spada u kategoriju **B - manje zahtevni objekti**. \n\nKlasifikacioni broj je **121113**.
        Assessment: YES

        Context: Prilikom izgradnje stambenih, stambeno-poslovnih i poslovno-stambenih objekata moraju se predvideti mere za spre\\u010davanje \\u0161irenja po\\u017eara:\\r\\n\\r\\n1) objekti ve\\u0107e du\\u017eine ili prislonjeni objekti u nizu izdvajaju se po\\u017earno na du\\u017eini od najvi\\u0161e 50 m zidovima otpornim prema po\\u017earu, rebrima na fasadi i krovu koji izlaze iz gabarita objekta najmanje 0,5 m ili primenom drugih tehni\\u010dkih re\\u0161enja tako da se spre\\u010di \\u0161irenje po\\u017eara;\\r\\n\\r\\n2) izvo\\u0111enje rebra iz ta\\u010dke 1) ovog \\u010dlana koje spre\\u010dava prenos po\\u017eara preko fasade objekta ve\\u0107e du\\u017eine se mo\\u017ee izostaviti ako je rastojanje prozora na ravnoj fasadi ve\\u0107e od 1m;\\r\\n\\r\\n3) izvo\\u0111enje rebra na krovu objekata iz ta\\u010dke 1) ovog \\u010dlana mo\\u017ee se izostaviti ako je sve do visine krovnog pokriva\\u010da objekat podeljen zidom odgovaraju\\u0107e otpornosti prema po\\u017earu, a krovni pokriva\\u010d i njegova podkonstrukcija otporni prema po\\u017earu u pojasu od 5 m;\\r\\n\\r\\n4) objekti sa hodnicima du\\u017eine ve\\u0107e od 40 m moraju biti podeljeni pregradom i vratima otpornim prema po\\u017earu najmanje 60 min;\\r\\n\\r\\n5) u objektima stan se izdvaja od susednih stanova, poslovnih i drugih prostorija, zidovima i me\\u0111uspratnim konstrukcijama otpornim prema po\\u017earu 90 min;\\r\\n\\r\\n6) zajedni\\u010dke stanarske ostave, prostorije za deponiju sme\\u0107a i sl. moraju se izdvojiti u posebne po\\u017earne sektore sa elementima konstrukcije otpornim prema po\\u017earu 90 min.\\r\\n\\r\\n### \\u010clan 21\\r\\n\\r\\nPrilikom izgradnje hotela i drugih ugostiteljskih objekata za sme\\u0161taj moraju se predvideti mere za spre\\u010davanje \\u0161irenja po\\u017eara:\\r\\n\\r\\n1) skladi\\u0161ne, tehni\\u010dke i prate\\u0107e prostorije (perionica, su\\u0161ionica i peglarnica ve\\u0161a, hladnja\\u010da i sl.) moraju se izdvojiti u posebne po\\u017earne sektore;\\r\\n\\r\\n2) deo objekta u kome se nalaze sobe odnosno apartmani, a u kojima boravi najvi\\u0161e 100 lica, mora se izdvojiti u poseban po\\u017earni sektor;\\r\\n\\r\\n3) za objekte visine H \\u2265 22 m sme\\u0161tajni deo hotela mora se izdvojiti u poseban po\\u017earni sektor uva\\u017eavaju\\u0107i zahteve iz ta\\u010dke 2) ovog \\u010dlana. Unutra\\u0161nja stepeni\\u0161ta ovog dela objekta moraju biti odvojena od hodnika zidovima i vratima otpornim prema po\\u017earu kao elementi na granici po\\u017earnog sektora, a u skladu sa projektovanim stepenom otpornosti prema po\\u017earu objekta.\\n\\nGornje prostorije dupleks stanova, slikarski ateljei i sli\\u010dne prostorije koje se manje koriste smatraju se prostorijama namenjenim za boravak ljudi. Stanarske ostave, podstanice grejanja, prostorije za ventilacione i klima komore, liftovske ku\\u0107ice i sl. ne smatraju se prostorijama za boravak ljudi;\\r\\n\\r\\n5) stambeni, odnosno stambeno-poslovni objekat jeste objekat koji ima vi\\u0161e od 80% korisne povr\\u0161ine namenjene za stanovanje i prate\\u0107e sadr\\u017eaje (prostorije za gara\\u017eiranje vozila, stanarske ostave, podstanicu grejanja, hodnike, stepeni\\u0161ta, liftovska okna i sl.) i u kome ostatak \\u010dini poslovni prostor;\\r\\n\\r\\n6) poslovni, odnosno poslovno-stambeni objekat jeste objekat koji ima vi\\u0161e od 80% korisne povr\\u0161ine namenjene za poslovanje (prodaju, kupovinu, usluge i sl.) i u kome su ostale prostorije stambene i/ili javne namene;\\r\\n\\r\\n7) objekat javne namene (u daljem tekstu: javni objekat) jeste objekat koji ima vi\\u0161e od 80% korisne povr\\u0161ine namenjene za okupljanje i boravak ljudi (ugostiteljski objekti za sme\\u0161taj - izuzev kampa, ku\\u0107e i stana za odmor, soba za iznajmljivanje, ugostiteljski objekti za ishranu i pi\\u0107e - izuzev kioska i sli\\u010dnih objekata, obrazovni, zdravstveni, kulturni, sportski, religiozni i sli\\u010dni objekti, autobuski, \\u017eelezni\\u010dki, avio-terminali, putni\\u010dka pristani\\u0161ta i sl.) i u kome su ostale prostorije stambene i/ili poslovne namene;\\r\\n\\r\\n8) izdvojeni objekat je objekat koji je od susednog objekta udaljen vi\\u0161e od 4 m;\\r\\n\\r\\n9) objekti u nizu su susedni objekti \\u010dije rastojanje iznosi manje od 4 m;\\r\\n\\r\\n10) po\\u017earni segment je deo objekta koji konstrukciono i funkcionalno \\u010dini jednu gra\\u0111evinsku celinu koja je i po\\u017earno izdvojena od ostalih\\r\\ndelova objekta gra\\u0111evinskim konstrukcijama otpornim prema po\\u017earu;\\r\\n\\r\\n11) po\\u017earni sektor je prostorija ili skup prostorija objekta koja se mo\\u017ee samostalno tretirati u pogledu nekih tehni\\u010dkih i organizacionih mera bezbednosti od po\\u017eara (po\\u017earno i specifi\\u010dno po\\u017earno optere\\u0107enje, zona dojave po\\u017eara, zona automatskog ga\\u0161enja po\\u017eara i sl.\\n\\n), a odvojena je od ostalih delova objekta gra\\u0111evinskim konstrukcijama otpornim prema po\\u017earu;\\r\\n\\r\\n12) stepen otpornosti objekta prema po\\u017earu je ocena pona\\u0161anja objekta na dejstvo po\\u017eara i izra\\u017eava se ocenama od I do V, odnosno\\r\\nneznatan (I), mali (II), srednji (III), ve\\u0107i (IV) i veliki (V);\\r\\n\\r\\n13) evakuacija je udaljavanje lica od polaznog do bezbednog mesta u slu\\u010daju opasnosti;\\r\\n\\r\\n14) polazno mesto (PM) je mesto na kome se mo\\u017ee zate\\u0107i lice u trenutku saznanja da je do\\u0161lo do takvog razvoja po\\u017eara da je potrebna evakuacija;\\r\\n\\r\\n15) bezbedno mesto (BM) je mesto van objekta udaljeno od izlaza iz objekta, na kome se ne mogu o\\u010dekivati \\u0161tetni efekti po\\u017eara (plamen, dim, pad o\\u0161te\\u0107enih delova objekta i sl.) ili posebno mesto u objektu izgra\\u0111eno u skladu sa propisom kojim je ure\\u0111ena oblast bezbedne evakuacije lica;\\r\\n\\r\\n16) prvi izlaz (PI) je izlaz iz prostorije ili grupe prostorija za boravak ka hodniku. To je obi\\u010dno izlaz iz stana, hotelskog apartmana ili\\r\\nsli\\u010dne grupe prostorija, u\\u010dionice, kancelarije i sl.\\r\\n\\r\\n17) eta\\u017eni izlaz (EI) \\u010dine vrata na izlazu iz hodnika otporna prema po\\u017earu ili vrata otporna prema po\\u017earu postavljena na ulazu u\\r\\nstepeni\\u0161te ili u hol;\\r\\n\\r\\n18) krajnji izlaz (KI) je izlaz iz objekta namenjen za evakuaciju;\\r\\n\\r\\n19) brzina evakuacije (ve) je projektna vrednost brzine kretanja lica kroz evakuacioni put;\\r\\n\\r\\n20) vreme evakuacije je vreme pripreme za evakuaciju i vreme kretanja od polaznog mesta do bezbednog mesta;\\r\\n\\r\\n21) progresivna horizontalna evakuacija je evakuacija koja se izvodi u objektima koji su projektovani i izvedeni na na\\u010din kojim se obezbe\\u0111uje proces evakuacije lica u susedni po\\u017earni sektor na istoj eta\\u017ei, a u kome su oni za\\u0161ti\\u0107eni i odakle se mogu evakuisati do bezbednog mesta;\\r\\n\\r\\n22) sigurnosna rasveta je ve\\u0161ta\\u010dko osvetljenje objekta ili prostora ili njihovog dela, dodata op\\u0161toj rasveti radi ispunjenja sigurnosnih uslova evakuacije ili bezbednog zavr\\u0161etka neke radne aktivnosti i automatski se uklju\\u010duje u slu\\u010daju kvara ili prekida u napajanju elektri\\u010dnom energijom op\\u0161te rasvete.\\r\\n\\r\\n## III PRISTUP ZA VATROGASNA VOZILA\\r\\n\\r\\n### \\u010clan 5\\r\\n\\r\\nSvakom stambenom, poslovnom i javnom objektu mora se obezbediti pristupni put izgra\\u0111en u skladu sa propisom kojim je ure\\u0111ena ova oblast.\\r\\n\\r\\nPristupni put iz stava 1. ovog \\u010dlana mora biti uvek slobodan i na njemu nije dozvoljeno parkiranje i zaustavljanje drugih vozila, niti postavljanje bilo kojih drugih prepreka koje ometaju vatrogasnu intervenciju.\\n\\nUkupna povr\\u0161ina poda prostorije ne obuhvata ni\\u0161e koje se koriste za sme\\u0161taj kabaste opreme (npr. pe\\u0107nica za pice, fri\\u017eidera,zamrziva\\u010da, ormana i regala), kao ni povr\\u0161ine poda namenjene za slaganje u visinu ve\\u0107u od 1 m.\\r\\n\\r\\n
        Query: Kako se stan izdvaja od susednih stanova i prostorija po pitanju sigurnosti od požara?
        Response: U objektima stan se izdvaja od susednih stanova, poslovnih i drugih prostorija, zidovima i međuspratnim kon­strukcijama otpornim prema požaru 90 min.
        Assessment: YES

        Context: ## II. USLOVI I NORMATIVI ZA PROJEKTOVANjE STAMBENIH ZGRADA I STANOVA\r\n\r\n### 1. Parkiranje vozila i garaža\r\n\r\n#### Član 3.\r\n\r\nParkiranje vozila za potrebe stambene zgrade, projektuje se u skladu sa planskim dokumentom, kao podzemni, nadzemni otvoreni ili natkriveni parking prostor, u okviru garaže ili kombinovano.\r\n\r\nBroj parking mesta po jednom stanu određuje se u skladu sa planskim dokumentom.\r\n\r\nPod garažom u stambenoj zgradi podrazumeva se zatvoreni prostor za parkiranje vozila.\r\n\r\nNajmanja dimenzija parking mesta za parkiranje je 230/480 cm, parking mesta za podužno parkiranje je 200/550 cm, a garažnog boksa 270/550 cm.\r\n\r\nSvetla visina garaža na površinama koje su predviđene za kretanje motornih vozila, ne sme biti manja od 220 cm, mereno od kote gotovog poda do najnižeg nivoa elemenata konstrukcije, instalacija i opreme.\r\n\r\nNajveći nagib pristupne rampe za garažu iznosi 12% ako je rampa otvorena, odnosno 15% ako je rampa zaštićena od zaleđivanja ili natkrivena.\r\n\r\n### 2. Pristupne površine\r\n\r\n#### Član 4.\r\n\r\nZa pristup pešaka i vozila zgradi planiraju se uređene površine.\r\n\r\nPešački pristup kod ulaza u zgradu mora biti minimalne širine 150 cm.\r\n\r\nKada se u zgradi planiraju i druge, nestambene namene (poslovanje, komercijalne delatnosti i sl), ulazni prostor namenjen ovim namenama mora biti odvojen od ulaza namenjenog stanovanju.\r\n\r\nUkoliko postoji denivelacija ulaza u zgradu u odnosu na pristup zgradi, pristup zgradi (stepenište, rampa i dr), kao i pristupnu površinu, treba planirati u skladu sa propisima o pristupačnosti.\r\n\r\nIzmeđu spoljašnjeg pristupnog stepeništa i vetrobrana neophodno je planirati ravnu površinu najmanje u širini stepeništa, minimalne dužine 120 cm.\r\n\r\n### 3. Prostori za kretanje u stambenoj zgradi\r\n\r\n#### Ulaz u zgradu\r\n\r\n##### Član 5.\r\n\r\nUlazni prostor u zgradu mora biti projektovan tako da je omogućen nadzor nad ulazom u objekat, komunikacija posetilaca sa stanarima (interfon) i dostava pošte.\r\n\r\nUlazni prostor mora biti projektovan tako da se u slučaju denivelacije pristupne površine i pristupne stanice lifta omogući nesmetano kretanje, u skladu sa propisima pristupačnosti.\r\n\r\nNa ulazu u stambenu zgradu mora se predvideti vetrobran. Dimenzije vetrobrana određuju se prema broju stanara koji ga koriste, kao i prema opremi. Najmanja dubina vetrobrana glavnog ulaza u zgradu iznosi 210 cm, a najmanja širina, u zgradi do 30 stanova iznosi 180 cm, u zgradi do 40 stanova iznosi 240 cm, a u zgradi sa 40 i više stanova iznosi 300 cm.\n\nSavladavanje visinske razlike između pešačke površine i prilaza do objekta vrši se:\r\n\r\n1. rampama za pešake i korisnike invalidskih kolicima, za visinsku razliku do 76 cm;\r\n2. spoljnim stepenicama, rampama, stepeništem i podiznim platformama, za visinsku razliku veću od 76 cm.\r\n\r\n#### Rampe za pešake\r\n\r\n##### Član 7\r\n\r\nSavladavanje visinskih razlika do 76 cm između dve pešačke površine i na prilazu do objekta vrši se primenom rampi tako da:\r\n\r\n1. nagib rampe nije veći od 5% (1:20), a ako nema uslova za rampu nagiba od 5% može iznositi 8.3% (1:12) za kratka rastojanja (do 6 m);\r\n2. najveća dozvoljena ukupna dužina rampe u posebnom slučaju iznosi 15 m;\r\n3. rampe duže od 6 m, a najviše do 9 m u slučaju da su manjeg nagiba, razdvajaju se odmorištima najmanje dužine 150 cm;\r\n4. najmanja čista širina rampe za jednosmeran prolaz iznosi 90 cm, a ukoliko je dvokraka čista širina rampe iznosi min 150 cm, sa podestom od min.\n\nZa savladavanje visinske razlike između trotoara i kolovoza koriste se zakošeni ivičnjaci koji se izvode u širini pešačkog prelaza i u nivou kolovoza, sa maksimalnim nagibom zakošenog dela do 8,3%, a ako je tehnički neizvodljivo u izuzetnim slučajevima do 10%.\r\n\r\nPovršina prolaza kroz pešačko ostrvo izvodi se sa taktilnim poljem bezbednosti/upozorenja, na celoj površini prolaza kroz ostrvo.\r\n\r\nZakošeni deo pešačke staze na mestu prelaza na kolovoz jednak je širini pešačkog prelaza.\r\n\r\nPovršina zakošenog dela pešačke staze na mestu prelaza na kolovoz izvedena je sa taktilnim poljem bezbednosti/upozorenja - Slika 18. Pešački prelazi i pešačka ostrva.\r\n\r\n#### Mesta za parkiranje\r\n\r\n##### Član 36\r\n\r\nMesta za parkiranje vozila koja koriste osobe sa invaliditetom predviđaju se u blizini ulaza u stambene zgrade, objekata za javno korišćenje i drugih objekata, označavaju se znakom pristupačnosti.\r\n\r\nMesta za parkiranje treba da ispune sledeće uslove:\r\n\r\n1. najmanja ukupna površina mesta za parkiranje vozila koja koriste osobe sa invaliditetom iznosi 370 cm x 480 cm;\r\n2. mesto za parkiranje za dva automobila koje se nalazi u nizu parkirališnih mesta upravno na trotoar veličine je 590 x 500 cm sa međuprostorom širine 150 cm;\r\n3. ako parkiralište nije izvedeno u istom nivou sa obližnjom pešačkom stazom tada se izlaz sa parkirališta obezbeđuje spuštenom pešačkom stazom maksimalnog nagiba od 8,3% i minimalne širine najmanje 140 cm koliko iznosi slobodan prostor za manevrisanje;\r\n4. pristupačno parking mesto mora uvek da se projektuje u horizontalnom položaju, a nikada na uzdužnom nagibu. Dozvoljen je samo odlivni poprečni nagib od maksimalno 2%;\r\n5. pristupačno parking mesto treba da ima direktnu pešačku vezu između projektovanog slobodnog prostora za manevar i najbliže pešačke staze, bez izlaska na kolovoz, u skladu sa preporukama datim za pešačke staze;\r\n6. potrebno je obezbediti pristupačan pešački pristup automatima ili šalteru za prodaju parking karata, u skladu sa preporukama za šaltere i uređaje u objektima.\n\nDozvoljen je samo odlivni poprečni nagib od maksimalno 2%;\r\n5. pristupačno parking mesto treba da ima direktnu pešačku vezu između projektovanog slobodnog prostora za manevar i najbliže pešačke staze, bez izlaska na kolovoz, u skladu sa preporukama datim za pešačke staze;\r\n6. potrebno je obezbediti pristupačan pešački pristup automatima ili šalteru za prodaju parking karata, u skladu sa preporukama za šaltere i uređaje u objektima.\r\n\r\nPored uslova iz stava 2. ovog člana predviđa se:\r\n1. za javne garaže, javna parkirališta, kao i parkirališta uz objekte za javno korišćenje i stambene i stambeno poslovne zgrade sa deset i više stanova, najmanje 5% mesta od ukupnog broja mesta za parkiranje, a najmanje jedno mesto za parkiranje vozila osoba sa invaliditetom;\r\n2. na parkiralištima sa manje od 20 mesta koja se nalaze uz ambulantu, apoteku, prodavnicu prehrambenih proizvoda, poštu, restoran i dečji vrtić, najmanje jedno mesto za parkiranje vozila osoba sa invaliditetom;\r\n3. na parkiralištima uz benzinske pumpe, restorane i motele pored magistralnih i regionalnih puteva 5% mesta od ukupnog broja mesta za parkiranje, ali ne manje od jednog mesta za parkiranje vozila osoba sa invaliditetom;\r\n4. na parkiralištima uz domove zdravlja, bolnice, domove starih i druge zdravstvene i socijalne ustanove, najmanje 10% mesta od ukupnog broja mesta za parkiranje, a najmanje dva mesta za parkiranje vozila osoba sa invaliditetom;\r\n5. svako parkiralište koje je obeleženo mora imati najmanje jedno pristupačno mesto za parkiranje.\r\n\r\n#### Stajališta javnog prevoza\r\n\r\n##### Član 37\r\n\r\nNa stajalištima javnog prevoza, predviđa se plato (peron) za pešake širine najmanje 300 cm, a prilazne pešačke staze treba da budu izvedene u istom nivou, bez denivalacija, prema preporukama za pešačke staze u skladu sa članom 32. ovog pravilnika.\r\n\r\nUkoliko plato stajališta javnog prevoza nije u istom nivou sa pešačkom stazom, pristup platformi obezbediće se spuštanjem staze ili platforme maksimalnog nagiba od 10%, ili pomoću rampe maksimalnog nagiba 5%, minimalne širine od 120 cm.\r\n\r\nZona ulaska u vozilo javnog prevoza ispred prednjih vrata vozila vizuelno se obeležava kontrastom i izvodi se taktilnim poljem bezbednosti minimalne površine 90 x 90 cm koje je povezano sa sistemom taktilne linije vođenja.\r\n\r\nUnutar objekata za javni transport (drumski, avio, železnički i drugi transport) ukoliko nije drugačije predviđeno, primenjuju se međunarodni standardi koji uređuju ovu oblast (IATA, UIC, ICAO i dr.).\n\nNajviši poprečni nagib uličnih trotoara i pešačkih staza upravno na pravac kretanja iznosi 2%.\r\n\r\nŠetališta u okviru javnih zelenih i rekreativnih površina su osvetljena, označena i sa obezbeđenim mestima za odmor sa klupama duž pravaca kretanja.\r\n\r\nKlupe treba da imaju sedišta na visini od 45 cm i rukohvate na visini od 70 cm iznad nivoa šetne staze u 50% od ukupnog broja klupa. Pored klupa se obezbeđuje prostor površine 110 cm sa 140 cm za smeštaj pomagala za kretanje.\r\n\r\n##### Član 33\r\n\r\nRadi nesmetanog kretanja širina uličnih trotoara i pešačkih staza iznosi 180 cm, a izuzetno 120 cm, dok širina prolaza između nepokretnih prepreka iznosi najmanje 90 cm.\r\n\r\nPovršina šetališta je čvrsta, ravna i otporna na klizanje. Profili rešetki, poklopaca i šahtova treba da budu bezbedni za kretanje učesnika u saobraćaju.\r\n\r\nNa trgovima ili na drugim velikim pešačkim površinama, kontrastom boja i materijala obezbeđuje se uočljivost glavnih tokova i njihovih promena u pravcu.\r\n\r\n##### Član 34\r\n\r\nU koridoru osnovnih pešačkih kretanja ne postavljaju se stubovi, reklamni panoi ili druge prepreke, a postojeće prepreke se vidno obeležavaju.\r\n\r\nDelovi zgrada kao što su balkoni, erkeri, viseći reklamni panoi i sl., kao i donji delovi krošnji drveća, koji se nalaze neposredno uz pešačke koridore, uzdignuti su najmanje 250 cm u odnosu na površinu po kojoj se pešak kreće.\r\n\r\n#### Pešački prelazi i pešačka ostrva\r\n\r\n
        Query: Koji je maksimalni nagib pesacke rampe?
        Response: Maksimalni nagib pešačke rampe zavisi od njene dužine i visinske razlike koju treba da savlada. Prema **Pravilniku o tehničkim standardima planiranja, projektovanja i izgradnje objekata kojima se obezbeđuje nesmetano kretanje i pristup osoba sa invaliditetom, dece i starih osoba ("Sl. glasnik RS", br. 44/2010)** : * **Za rampe dužine do 6 metara**, maksimalni nagib je **8,3% (1:12)**. Ovaj nagib se primenjuje kada nije moguće obezbediti rampu sa nagibom od 5%. * **Za rampe dužine preko 6 metara**, maksimalni nagib je **5% (1:20)**. Dodatno: * **Ukupna dužina rampe** u posebnom slučaju ne sme biti veća od **15 metara**. * **Rampe duže od 6 metara, a najviše do 9 metara**, u slučaju da su manjeg nagiba, moraju se razdvajati **odmorištima** minimalne dužine **150 cm**. Ukoliko je visinska razlika veća od 76 cm, neophodno je predvideti kombinaciju rampe sa stepeništem i/ili platformom za podizanje.
        Assessment: YES
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
        llama_index_transformed_history = self._format_chat_history(transformed_history[-5:])

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
