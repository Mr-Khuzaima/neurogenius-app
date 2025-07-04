from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, initialize_agent
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
import hashlib
import os
from typing import Dict, List

class Config:
    def _init_(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.persist_dir = "./chroma_data"
        os.makedirs(self.persist_dir, exist_ok=True)

config = Config()

class AIService:
    def _init_(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        self.embeddings = OpenAIEmbeddings()
        self.recognizer = sr.Recognizer()
        self.search_tool = DuckDuckGoSearchRun()
        
        self.tools = [
            Tool(name="Web Search", func=self.search_tool.run, 
                description="Useful for finding current information"),
            Tool(name="Document QA", func=self.query_document,
                description="For answering questions about documents")
        ]
        self.agent = initialize_agent(self.tools, self.llm, agent="zero-shot-react-description")
    
    def _doc_id(self, file_bytes: bytes) -> str:
        return hashlib.md5(file_bytes).hexdigest()
    
    def process_pdf(self, file_bytes: bytes) -> str:
        doc_id = self._doc_id(file_bytes)
        doc_path = os.path.join(config.persist_dir, doc_id)
        
        if os.path.exists(os.path.join(doc_path, "chroma.sqlite3")):
            return doc_id
        
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            loader = PyPDFLoader(tmp.name)
            pages = loader.load()
            os.unlink(tmp.name)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)
        
        Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=doc_path
        )
        return doc_id
    
    def query_document(self, doc_id: str, question: str) -> str:
        doc_path = os.path.join(config.persist_dir, doc_id)
        if not os.path.exists(doc_path):
            return "Document not found."
        
        vectorstore = Chroma(persist_directory=doc_path, embedding_function=self.embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        return qa.run(question)
    
    def web_search(self, query: str) -> str:
        return self.search_tool.run(query)
    
    def summarize_text(self, doc_id: str) -> str:
        doc_path = os.path.join(config.persist_dir, doc_id)
        if not os.path.exists(doc_path):
            return "Document not found."
            
        vectorstore = Chroma(persist_directory=doc_path, embedding_function=self.embeddings)
        docs = vectorstore.similarity_search("", k=10)
        combined = "\n\n".join([d.page_content for d in docs])
        
        prompt = """Summarize in 3-5 key points: {text}"""
        llm_chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(prompt))
        return llm_chain.run(text=combined)
    
    def extract_tables(self, doc_id: str) -> List[Dict]:
        doc_path = os.path.join(config.persist_dir, doc_id)
        if not os.path.exists(doc_path):
            return []
            
        vectorstore = Chroma(persist_directory=doc_path, embedding_function=self.embeddings)
        docs = vectorstore.similarity_search("table", k=5)
        
        tables = []
        for doc in docs:
            if "|" in doc.page_content or "\t" in doc.page_content:
                tables.append({"content": doc.page_content, "source": doc.metadata.get("source", "")})
        return tables
    
    def translate_text(self, text: str, target_language: str) -> str:
        prompt = f"Translate to {target_language}:\n{text}"
        return self.llm.predict(prompt)
    
    def transcribe_audio(self, audio_bytes: bytes) -> str:
        try:
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
            wav_data = audio.set_frame_rate(16000).set_channels(1).export(format="wav").read()
            
            with sr.AudioFile(BytesIO(wav_data)) as source:
                audio_data = self.recognizer.record(source)
                return self.recognizer.recognize_google(audio_data)
        except Exception as e:
            raise HTTPException(500, f"Transcription error: {str(e)}")

app = FastAPI()
ai = AIService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    return {"doc_id": ai.process_pdf(await file.read())}

@app.post("/query")
async def query(payload: dict):
    return {"response": ai.query_document(payload["doc_id"], payload["question"])}

@app.post("/search")
async def search(payload: dict):
    return {"results": ai.web_search(payload["query"])}

@app.post("/summarize")
async def summarize(payload: dict):
    return {"summary": ai.summarize_text(payload["doc_id"])}

@app.post("/tables")
async def tables(payload: dict):
    return {"tables": ai.extract_tables(payload["doc_id"])}

@app.post("/translate")
async def translate(payload: dict):
    return {"translation": ai.translate_text(payload["text"], payload["target_language"])}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    return {"text": ai.transcribe_audio(await file.read())}
