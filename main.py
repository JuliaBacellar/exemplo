from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline  # Importe da nova localização
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

#
CAMINHO_MODELO = "pablocosta/bertabaporu-gpt2-small-portuguese"  # GPT-2 em PT

def carregar_llm_local():
    tokenizer = AutoTokenizer.from_pretrained(CAMINHO_MODELO)
    model = AutoModelForCausalLM.from_pretrained(CAMINHO_MODELO)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    return HuggingFacePipeline(pipeline=pipe)

# Carrega e divide o PDF
def carregar_documentos():
    loader = PyPDFLoader("data/exemplo.pdf")
    documentos = loader.load()
    splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents(documentos)


# Cria ou carrega os vetores
def carregar_ou_criar_vectorstore(documentos):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # Modelo mais leve
    )
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documentos, embeddings)
        vectorstore.save_local("faiss_index")
        return vectorstore
# Função principal
def main():
    print("Carregando... - Digite 'sair' para encerrar\n")

    documentos = carregar_documentos()
    vectorstore = carregar_ou_criar_vectorstore(documentos)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = carregar_llm_local()

    prompt = PromptTemplate.from_template("""
    <|system|>
    Você é um assistente que responde em português com base no contexto fornecido.</s>
    <|user|>
    Pergunta: {query}</s>
    <|assistant|>
    Resposta:""")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    while True:
        pergunta = input("Pergunta: ")
        if pergunta.lower() == "sair":
            break

        resposta = qa.invoke({"query": pergunta})
        print("\nResposta:", resposta["result"], "\n")


if __name__ == "__main__":
    main()
