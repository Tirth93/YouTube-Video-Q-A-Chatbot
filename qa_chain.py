from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # <- Use this instead of OpenLLM
import os

def build_qa_chain(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)

    llm = ChatOpenAI(
        model_name="mistral",  # You can name it anything if LM Studio is listening
        base_url=os.getenv("LOCAL_LLM_ENDPOINT"),  # This is key
        temperature=0.7,
        api_key="lm-studio",  # Dummy value required by LangChain
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain
