from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA  
import os


CAMINHO_MODELO = "modelo/mistral-7b-instruct-v0.1.Q4_K_M.gguf"



# Carrega e divide o PDF
def carregar_documentos():
    loader = PyPDFLoader("data/exemplo.pdf")
    documentos = loader.load()
    splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents(documentos)

# Cria ou carrega os vetores
def carregar_ou_criar_vectorstore(documentos):
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documentos, HuggingFaceEmbeddings())
        vectorstore.save_local("faiss_index")
        return vectorstore

# Função principal
def main():
    print("Carregando... - Digite 'sair' para encerrar\n")

    documentos = carregar_documentos()
    vectorstore = carregar_ou_criar_vectorstore(documentos)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

   
    llm = CTransformers(
        model= CAMINHO_MODELO,
        model_type= "mistral", 
        config= { 
            'max_new_tokens': 400,
            'context_length': 4096,
            'temperature': 0.7,
            'top_p': 0.95,
            'repetition_penalty': 1.1,
            'threads': 3,
            'context_length': 2048,
        }
    )

    prompt = PromptTemplate.from_template("""
    Você é um assistente útil e sempre responde em português, com base nos documentos fornecidos.

    Pergunta: {query}
    Resposta:
    """)

    qa = RetrievalQA.from_chain_type (
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    while True:
        pergunta = input("Pergunta: ")
        if pergunta.lower() == "sair":
            break

        resposta = qa.invoke({"query": pergunta})
        print("\nResposta:", resposta["result"], "\n")


if __name__ == "__main__":
    main()
