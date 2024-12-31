import os
import warnings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageDocumentParseLoader
import re

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain.docstore.document import Document

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# warnings.filterwarnings("ignore")
def clean_parsed_text(text):
    # 이스케이프 시퀀스 제거
    text = text.replace('\\"', '"').replace('\\n', '\n')
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # HTML 태그 제거 (선택적)
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()

# layzer.lazy_load()
# 메모리 효율을 향상시키기 위해서, lazy_load() 로 페이지별로 문서를 불러올 수도 있음

#### Text Split

def loadDocs() :
    # .env 파일 로드
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    #### SET API KEY

    # Upstae API KEY
    os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY")
    # Pinecone API KEY
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")

    #### Document Loading


    layzer = UpstageDocumentParseLoader(
        "sqld_summary.pdf", # 불러올 파일
        output_format='html',  # 결과물 형태 : HTML
        coordinates= False) # 이미지 OCR 좌표계 가지고 오지 않기

    docs = layzer.load()
    return docs


def getSplit(docs) :
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100)

    splits = text_splitter.split_documents(docs)

    for document in splits :
      document.page_content = clean_parsed_text(document.page_content)

    # print("Splits:", len(splits))
    return splits


#### VECTOR STORE

def getVectorStore(splits) :
    # 3. Embed & indexing
    vectorstore = PineconeVectorStore.from_documents(
        splits, UpstageEmbeddings(model="embedding-query"), index_name="retriever-demo"
    )
    return vectorstore


def getRetriever(vectorstore) :
    retriever = vectorstore.as_retriever(
    search_type= 'mmr', # default : similarity(유사도) / mmr 알고리즘
    search_kwargs={"k": 3} # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
    )
    return retriever

#### 연습문제 출력하기위해선 Prompt 수정 필요요
def getChain() :
    #### Creating a Prompt with Retrieved Result
    llm = ChatUpstage()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question considering the history of the conversation.
                If you don't know the answer, just say that you don't know.
                ---
                CONTEXT:
                {context}
                """,
            ),
            ("human", "{input}"),
        ]
    )

    #### Implementing an LLM Chain
    chain = prompt | llm | StrOutputParser()
    return chain


