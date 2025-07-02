from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
import os
import time
import traceback

# Configurações globais
CONFIG = {
    "model_path": "modelo/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "pdf_path": "data/exemplo.pdf",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "faiss_index": "faiss_index",
    "llm_config": {
        'max_new_tokens': 150,
        'temperature': 0.7,
        'gpu_layers': 40 if os.environ.get("CUDA_VISIBLE_DEVICES") else 0,
        'batch_size': 8,
        'threads': min(4, os.cpu_count()),
        'context_length': 2048
    }
}

def inicializar_sistema():
    """Configura cache e verifica recursos"""
    set_llm_cache(InMemoryCache())
    print(f"🖥️ Threads disponíveis: {CONFIG['llm_config']['threads']}")
    print(f"🏗️ Configuração GPU: {'Ativada' if CONFIG['llm_config']['gpu_layers'] else 'Desativada'}")

def carregar_documentos():
    """Carrega e divide o documento PDF"""
    print("⏳ Carregando documento...")
    start = time.time()
    
    if not os.path.exists(CONFIG["pdf_path"]):
        raise FileNotFoundError(f"Arquivo PDF não encontrado em {CONFIG['pdf_path']}")

    loader = PyPDFLoader(CONFIG["pdf_path"])
    documentos = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    documentos_split = splitter.split_documents(documentos)
    print(f"✅ Documento carregado em {time.time() - start:.2f}s")
    print(f"📄 Páginas: {len(documentos)} | Chunks: {len(documentos_split)}")
    return documentos_split

def inicializar_vectorstore(documentos):
    """Inicializa ou carrega o vectorstore FAISS"""
    print("⏳ Preparando vetores...")
    start = time.time()
    
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={'device': 'cuda' if CONFIG['llm_config']['gpu_layers'] else 'cpu'}
    )
    
    if os.path.exists(CONFIG["faiss_index"]):
        vectorstore = FAISS.load_local(
            CONFIG["faiss_index"], 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"🔍 Vectorstore carregado: {vectorstore.index.ntotal} vetores")
    else:
        vectorstore = FAISS.from_documents(documentos, embeddings)
        vectorstore.save_local(CONFIG["faiss_index"])
        print("🆕 Novo vectorstore criado")
    
    print(f"✅ Vetores prontos em {time.time() - start:.2f}s")
    return vectorstore

def criar_llm():
    """Configura o modelo de linguagem"""
    print("🔥 Aquecendo o modelo LLM...")
    start = time.time()
    
    llm = CTransformers(
        model=CONFIG["model_path"],
        model_type="llama",
        config=CONFIG["llm_config"]
    )
    
    # Teste inicial
    llm.invoke("Warming up")
    print(f"✅ Modelo pronto em {time.time() - start:.2f}s")
    return llm

def criar_chain(llm, retriever):
    """Configura a cadeia de QA"""
    template = """Você é um assistente AI útil. Responda em português de forma clara e concisa.

Contexto: {context}
Pergunta: {question}

Resposta:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )


def processar_pergunta(qa_chain, pergunta):
    """Processa uma pergunta e exibe o resultado"""
    print("⏳ Processando...")
    start = time.time()
    
    try:
        resposta = qa_chain.invoke({"question": pergunta})
        tempo_resposta = time.time() - start
        print(f"\n🧠 Resposta ({tempo_resposta:.2f}s):")
        print(resposta["result"].strip() + "\n")
    except Exception as e:
        print("\n⚠️ Erro na geração da resposta:")
        traceback.print_exc()
        print(f"\nDica: Tente reformular sua pergunta.\n")

def main():
    """Função principal do sistema de QA"""
    print("\n" + "="*50)
    print("🔄 Iniciando Sistema de Perguntas e Respostas")
    print("="*50 + "\n")
    
    try:
        inicializar_sistema()
        documentos = carregar_documentos()
        vectorstore = inicializar_vectorstore(documentos)
        llm = criar_llm()
        
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 2}
        )
        
        qa_chain = criar_chain(llm, retriever)
        
        print("\n" + "="*50)
        print("✅ Sistema pronto! Digite sua pergunta ou 'sair' para encerrar")
        print("="*50 + "\n")
        
        while True:
            pergunta = input("❓ Pergunta: ").strip()
            if pergunta.lower() in ('sair', 'exit', 'quit'):
                break
                
            if not pergunta:
                print("⚠️ Por favor, digite uma pergunta válida.\n")
                continue
                
            processar_pergunta(qa_chain, pergunta)
            
    except Exception as e:
        print("\n❌ Erro crítico durante a inicialização:")
        traceback.print_exc()
        print("\nSugestões:")
        print("- Verifique se o arquivo PDF existe")
        print("- Confira o caminho do modelo LLM")
        print("- Verifique os recursos disponíveis (GPU/RAM)\n")
    finally:
        print("\n🔴 Sistema encerrado. Até logo!\n")

if __name__ == "__main__":
    main()