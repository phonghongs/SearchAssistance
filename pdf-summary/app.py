import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import uuid
import boto3
import os
import fitz
import secrets
from PIL import Image
from flask import Flask, jsonify, render_template, request, session
from dotenv import load_dotenv
from flask.sessions import SecureCookieSession
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

# Add AWS credentials configuration
session_aws = boto3.Session(
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_DEFAULT_REGION')
)

# Create the embeddings with the session
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=session_aws.client('bedrock-runtime')
)

llm = ChatBedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0,
    max_tokens=None,
    client=session_aws.client('bedrock-runtime')
)


@dataclass
class ChatSession:
    chain: any
    chat_history: List[Tuple[str, str]]
    pdf_path: str
    current_page: int


class PDFProcessor:
    """Class to handle PDF processing operations"""
    def __init__(self, temp_dir: str = "temp_pdfs"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def save_pdf(self, file) -> str:
        file_path = self.temp_dir / f"temp_{secrets.token_hex(8)}.pdf"
        file.save(file_path)
        return str(file_path)

    def render_page(self, file_path: str, page_num: int) -> str:
        """Render a PDF page as base64 image"""
        doc = fitz.Document(file_path)
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old temporary PDF files"""
        current_time = time.time()
        for file_path in self.temp_dir.glob("*.pdf"):
            gap = (current_time - file_path.stat().st_mtime)
            if gap > max_age_hours * 3600:
                file_path.unlink()


class ChatManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}

    def create_chain(self, file_path: str) -> any:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=4
        )
        all_splits = text_splitter.split_documents(documents)

        vectordb = InMemoryVectorStore(embeddings)
        vectordb.add_documents(all_splits)

        prompt = self.create_prompt()

        retriever = vectordb.as_retriever()
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        return rag_chain

    def create_prompt(self):
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        return prompt

    def create_session(self, session_id: str, file_path: str):
        chain = self.create_chain(file_path)
        self.sessions[session_id] = ChatSession(
            chain=chain,
            chat_history=[],
            pdf_path=file_path,
            current_page=0
        )

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get existing chat session"""
        return self.sessions.get(session_id)

    def process_message(self, session_id: str, message: str) -> Tuple[str, int]:
        session = self.sessions[session_id]

        result = session.chain.invoke({"input": message})

        print(result)
        page_num = result['context'][0].metadata['page']

        result = result['answer']
        # Add the Q&A pair to chat history
        session.chat_history.append((message, result))
        session.current_page = page_num

        return result, page_num


class PDFChatApp:
    """Main application class"""
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = secrets.token_hex(16)
        self.pdf_processor = PDFProcessor()
        self.chat_manager = ChatManager()
        self.setup_routes()

    def setup_routes(self):
        """Set up Flask routes"""
        self.app.route('/')(self.index)
        self.app.route('/upload-pdf', methods=['POST'])(self.upload_pdf)
        self.app.route('/chat', methods=['POST'])(self.chat)

    def index(self):
        """Render the main page"""
        session['id'] = str(uuid.uuid4())
        return render_template('index.html')

    def upload_pdf(self):
        print(session.get('id'))

        """Handle PDF upload"""
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.pdf'):
            return jsonify({'error': 'Invalid file'}), 400

        # Save PDF and create chat session
        file_path = self.pdf_processor.save_pdf(file)

        self.chat_manager.create_session(session.get('id'), file_path)

        # Render first page
        img_data = self.pdf_processor.render_page(file_path, 0)
        return jsonify({
            'status': 'success',
            'image': img_data
        })

    def chat(self):
        """Handle chat messages"""
        message = request.json.get('message')
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        chat_session = self.chat_manager.get_session(session.get('id'))
        if not chat_session:
            return jsonify({'error': 'No active session'}), 400

        # Process message and get response
        response, page_num = self.chat_manager.process_message(session.get('id'), message)

        # Render new page
        img_data = self.pdf_processor.render_page(chat_session.pdf_path, page_num)

        return jsonify({
            'response': response,
            'image': img_data
        })

    def run(self, **kwargs):
        """Run the Flask application"""
        self.app.run(**kwargs)


def create_app():
    """Application factory function"""
    app = PDFChatApp()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
