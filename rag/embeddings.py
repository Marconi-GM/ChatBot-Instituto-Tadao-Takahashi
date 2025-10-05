from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY, EMBED_MODEL
from langchain_community.embeddings import HuggingFaceEmbeddings



def get_retriever(docs, chunk_size=300, overlap=30):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)

    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model=EMBED_MODEL,
    #     google_api_key=GOOGLE_API_KEY
    #     )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold":0.3, "k":4}
    )
