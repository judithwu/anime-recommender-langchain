import pickle
from langchain.document_loaders import DataFrameLoader
from apikey import OPENAI_API_KEY, PINECONE_API_KEY
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone.index import Index
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

#load in dataframe
with open('dataframe.pkl', 'rb') as file:
    anime = pickle.load(file)

#load dataframe into docs
docs = DataFrameLoader(anime, page_content_column="page_content").load()

#initialize pinecone
pinecone.init(api_key = PINECONE_API_KEY, environment="gcp-starter")

index_name = "animes"
#check that the given index does not exist yet
if index_name not in pinecone.list_indexes():
    #create index if it does not exist
    pinecone.create_index(
        name = index_name,
        metric="cosine",
        dimension=1536
    )

#Fill the index with embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
index = Index(index_name)

#check if there is already data in the index on Pinecone
if index.describe_index_stats()['total_vector_count'] > 0:
    # use from_existing_index to use the vector store
    docsearch = Pinecone.from_existing_index(
        index_name,
        embeddings
    )
else:
    #if not, use from_documents to fill the vector store
   docsearch = Pinecone.from_documents(
        docs,
       embeddings,
       index_name = index_name
   )
    

#Create prompt templates 
DOCUMENT_PROMPT = """{page_content}
MyAnimeList link: {link}
========="""

QUESTION_PROMPT = """Given the following extracted parts of an anime database and a question, create a final answer with the link as source ("SOURCE").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCE" part in your answer.

QUESTION: What's a good anime about sports and school life?
=========
Title: Slam Dunk
Genre: Comedy, Drama, School, Shounen, Sports
Description: Hanamichi Sakuragi, infamous for his temper, massive height, and fire-red hair, enrolls in Shohoku High, hoping to finally get a girlfriend and break his record of being rejected 50 consecutive times in middle school. His notoriety precedes him, however, leading to him being avoided by most students. Soon, after certain events, Hanamichi is left with two unwavering thoughts: "I hate basketball," and "I desperately need a girlfriend." One day, a girl named Haruko Akagi approaches him without any knowledge of his troublemaking ways and asks him if he likes basketball. Hanamichi immediately falls head over heels in love with her, blurting out a fervent affirmative. She then leads him to the gymnasium, where she asks him if he can do a slam dunk. In an attempt to impress Haruko, he makes the leap, but overshoots, instead slamming his head straight into the blackboard. When Haruko informs the basketball team's captain of Hanamichi's near-inhuman physical capabilities, he slowly finds himself drawn into the camaraderie and competition of the sport he had previously held resentment for.
MyAnimeList link: https://myanimelist.net/anime/170
=========
Title: Haikyuu!!
Genre: Comedy, Sports, Drama, School, Shounen
Description: Inspired after watching a volleyball ace nicknamed "Little Giant" in action, small-statured Shouyou Hinata revives the volleyball club at his middle school. The newly-formed team even makes it to a tournament; however, their first match turns out to be their last when they are brutally squashed by the "King of the Court," Tobio Kageyama. Hinata vows to surpass Kageyama, and so after graduating from middle school, he joins Karasuno High School's volleyball teamâ€”only to find that his sworn rival, Kageyama, is now his teammate. Thanks to his short height, Hinata struggles to find his role on the team, even with his superior jumping power. Surprisingly, Kageyama has his own problems that only Hinata can help with, and learning to work together appears to be the only way for the team to be successful. Based on Haruichi Furudate's popular shounen manga of the same name, Haikyuu!! is an exhilarating and emotional sports comedy following two determined athletes as they attempt to patch a heated rivalry in order to make their high school volleyball team the best in Japan.
MyAnimeList link: https://myanimelist.net/anime/20583
=========
Title: Kuroko no Basket
Genre: Comedy, School, Shounen, Sports
Description: Teikou Junior High School's basketball team is crowned champion three years in a row thanks to five outstanding players who, with their breathtaking and unique skills, leave opponents in despair and fans in admiration. However, after graduating, these teammates, known as "The Generation of Miracles", go their separate ways and now consider each other as rivals. At Seirin High School, two newly recruited freshmen prove that they are not ordinary basketball players: Taiga Kagami, a promising player returning from the US, and Tetsuya Kuroko, a seemingly ordinary student whose lack of presence allows him to move around unnoticed. Although Kuroko is neither athletic nor able to score any points, he was a member of Teikou's basketball team, where he played as the "Phantom Sixth Man," who easily passed the ball and assisted his teammates. Kuroko no Basket follows the journey of Seirin's players as they attempt to become the best Japanese high school team by winning the Interhigh Championship. To reach their goal, they have to cross pathways with several powerful teams, some of which have one of the five players with godlike abilities, whom Kuroko and Taiga make a pact to defeat.
MyAnimeList link: https://myanimelist.net/anime/11771
=========
FINAL ANSWER: Kuroko no Basket is an anime about middle school basketball prodigies along with a sixth man that lurked in the shadows who helped the team earn their prestigious status. The team parts ways during high school, where the sixth man forms a dynamic partnership with a player who has significantly different abilities and opposite personality. Together, their goal is to conquer the high school basketball league, while appearances from former prodigy players complicate their plan.
SOURCE: https://myanimelist.net/anime/11771

QUESTION: {question}
=========
{summaries}
FINAL ANSWER:"""

#create prompt template objects
document_prompt = PromptTemplate.from_template(DOCUMENT_PROMPT)
question_prompt = PromptTemplate.from_template(QUESTION_PROMPT)


def generate_response(question):
    #create the QA LLM chain
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        chain_type="stuff",
        llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model_name = "gpt-3.5-turbo", temperature=0),
        chain_type_kwargs= {
            "prompt": question_prompt,
            "document_prompt": document_prompt
        },
        retriever = docsearch.as_retriever()
    )
    return qa_with_sources(question)
