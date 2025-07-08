from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

docs = [Document(page_content="The capital of Japan is Tokyo.")]
vectorstore = FAISS.from_documents(docs, embeddings)

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

result = qa_chain({"question": "What is the capital of Japan?"})
print("Answer:", result["answer"])
print("Sources:", result["sources"])
