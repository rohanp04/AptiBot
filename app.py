import os
import json
import time
import datetime
from typing import List, Dict, Any
from pathlib import Path
import tempfile

import streamlit as st
import fitz  # PyMuPDF
import docx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
import pandas as pd
from dotenv import load_dotenv
import hashlib

# Load environment variables
load_dotenv()

# ---------- Configuration ----------
TEMP_FOLDER = Path("temp")
TEMP_FOLDER.mkdir(exist_ok=True)

def load_documents_from_temp_folder() -> List[str]:
    """Load all documents from the temp folder."""
    temp_files = []
    supported_extensions = ['.pdf', '.docx', '.txt']
    
    for ext in supported_extensions:
        temp_files.extend(TEMP_FOLDER.glob(f"*{ext}"))
    
    return sorted([str(f) for f in temp_files])

# Initialize session state with enhanced quiz history tracking
def init_session_state():
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'embeddings_model' not in st.session_state:
        st.session_state.embeddings_model = None
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = []
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_start_time' not in st.session_state:
        st.session_state.quiz_start_time = None
    if 'timer_setting' not in st.session_state:
        st.session_state.timer_setting = 60
    if 'question_start_time' not in st.session_state:
        st.session_state.question_start_time = None
    
    # Enhanced session state for quiz history and question tracking
    if 'quiz_history' not in st.session_state:
        st.session_state.quiz_history = []
    if 'used_questions' not in st.session_state:
        st.session_state.used_questions = set()
    if 'current_quiz_id' not in st.session_state:
        st.session_state.current_quiz_id = None
    if 'all_generated_questions' not in st.session_state:
        st.session_state.all_generated_questions = []

# ---------- Question Management Functions ----------
def generate_question_hash(question_dict: Dict[str, Any]) -> str:
    """Generate a unique hash for a question to track duplicates."""
    question_text = question_dict.get('question', '')
    options_text = str(question_dict.get('options', {}))
    combined_text = f"{question_text}_{options_text}"
    return hashlib.md5(combined_text.encode()).hexdigest()

def filter_unused_questions(questions: List[Dict[str, Any]], num_needed: int) -> List[Dict[str, Any]]:
    """Filter out previously used questions and return required number of fresh questions."""
    unused_questions = []
    
    for question in questions:
        question_hash = generate_question_hash(question)
        if question_hash not in st.session_state.used_questions:
            unused_questions.append({**question, 'hash': question_hash})
    
    # If we don't have enough unused questions, reset and allow reuse
    if len(unused_questions) < num_needed:
        st.warning(f"⚠️ Only {len(unused_questions)} unused questions available. Including some previously used questions.")
        # Reset used questions if we've exhausted the pool
        if len(unused_questions) == 0:
            st.session_state.used_questions.clear()
            st.info("🔄 Question pool reset - all questions are now available again.")
            return filter_unused_questions(questions, num_needed)
        
        # Add some used questions back to meet the requirement
        used_questions = [q for q in questions if generate_question_hash(q) in st.session_state.used_questions]
        additional_needed = num_needed - len(unused_questions)
        additional_questions = used_questions[:additional_needed]
        
        for q in additional_questions:
            unused_questions.append({**q, 'hash': generate_question_hash(q)})
    
    return unused_questions[:num_needed]

def mark_questions_as_used(questions: List[Dict[str, Any]]):
    """Mark questions as used to prevent repetition."""
    for question in questions:
        if 'hash' in question:
            st.session_state.used_questions.add(question['hash'])
        else:
            question_hash = generate_question_hash(question)
            st.session_state.used_questions.add(question_hash)

# ---------- Quiz History Management ----------
def save_quiz_to_history(results: Dict[str, Any]):
    """Save completed quiz results to session history."""
    quiz_record = {
        'quiz_id': st.session_state.current_quiz_id,
        'completed_at': datetime.datetime.now().isoformat(),
        'results': results,
        'topic': getattr(st.session_state, 'current_topic', 'Unknown'),
        'total_questions': len(st.session_state.quiz_questions),
        'timer_setting': st.session_state.timer_setting
    }
    
    st.session_state.quiz_history.append(quiz_record)
    
    # Keep only last 10 quiz records to prevent memory bloat
    if len(st.session_state.quiz_history) > 10:
        st.session_state.quiz_history = st.session_state.quiz_history[-10:]

def get_quiz_statistics() -> Dict[str, Any]:
    """Calculate overall statistics from quiz history."""
    if not st.session_state.quiz_history:
        return {}
    
    total_quizzes = len(st.session_state.quiz_history)
    total_questions_answered = sum(int(h['results']['score'].split('/')[1]) for h in st.session_state.quiz_history if '/' in h['results']['score'])
    total_correct = sum(int(h['results']['score'].split('/')[0]) for h in st.session_state.quiz_history if '/' in h['results']['score'])
    
    avg_percentage = sum(h['results']['percentage'] for h in st.session_state.quiz_history) / total_quizzes
    best_score = max(h['results']['percentage'] for h in st.session_state.quiz_history)
    
    recent_trend = []
    if total_quizzes >= 3:
        recent_trend = [h['results']['percentage'] for h in st.session_state.quiz_history[-3:]]
    
    return {
        'total_quizzes': total_quizzes,
        'total_questions_answered': total_questions_answered,
        'total_correct': total_correct,
        'average_percentage': round(avg_percentage, 1),
        'best_score': round(best_score, 1),
        'recent_trend': recent_trend
    }

# ---------- Document Processing ----------
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n\n"
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF {file_path}: {str(e)}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX {file_path}: {str(e)}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading TXT {file_path}: {str(e)}")
        return ""

def process_uploaded_files(uploaded_files) -> str:
    """Process multiple uploaded files and extract text."""
    all_text = ""
    
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        temp_path = TEMP_FOLDER / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text based on file type
        file_ext = uploaded_file.name.lower().split('.')[-1]
        if file_ext == 'pdf':
            text = extract_text_from_pdf(str(temp_path))
        elif file_ext == 'docx':
            text = extract_text_from_docx(str(temp_path))
        elif file_ext == 'txt':
            text = extract_text_from_txt(str(temp_path))
        else:
            st.warning(f"Unsupported file format: {file_ext}")
            continue
        
        if text:
            all_text += f"\n\n--- Content from {uploaded_file.name} ---\n\n{text}"
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
    
    return all_text

def process_temp_folder_files() -> str:
    """Process all files in the temp folder and extract text."""
    all_text = ""
    temp_files = load_documents_from_temp_folder()
    
    if not temp_files:
        return ""
    
    for file_path in temp_files:
        file_name = Path(file_path).name
        file_ext = file_name.lower().split('.')[-1]
        
        # Extract text based on file type
        if file_ext == 'pdf':
            text = extract_text_from_pdf(file_path)
        elif file_ext == 'docx':
            text = extract_text_from_docx(file_path)
        elif file_ext == 'txt':
            text = extract_text_from_txt(file_path)
        else:
            st.warning(f"Unsupported file format: {file_ext}")
            continue
        
        if text:
            all_text += f"\n\n--- Content from {file_name} ---\n\n{text}"
    
    return all_text

def create_vector_store(text: str):
    """Create FAISS vector store from text."""
    if not text.strip():
        return None, None
    
    # Initialize embeddings model
    embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    if not chunks:
        return None, None
    
    # Create embeddings
    embeddings = embeddings_model.encode(chunks, normalize_embeddings=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    index.add(embeddings.astype('float32'))
    
    return index, chunks

def retrieve_context(query: str, index, chunks: List[str], embeddings_model, top_k: int = 3) -> str:
    """Retrieve relevant context for query."""
    if index is None or not chunks:
        return ""
    
    query_embedding = embeddings_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding.astype('float32'), top_k)
    
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return "\n\n".join(relevant_chunks)

# ---------- Question Generation ----------
def generate_questions(topic: str, num_questions: int, context: str = "") -> List[Dict[str, Any]]:
    """Generate aptitude questions using Groq."""
    if "GROQ_API_KEY" not in os.environ:
        st.error("GROQ_API_KEY not set as environment variable.")
        return []
    
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)
    
    context_prompt = f"\n\nContext from documents:\n{context}" if context else ""
    
    prompt = f"""Generate {num_questions} multiple choice aptitude questions on the topic: {topic}.
    
    {context_prompt}
    
    For each question, provide:
    1. A clear, challenging question
    2. Four options (A, B, C, D)
    3. The correct answer (A, B, C, or D)
    4. A detailed explanation of why the answer is correct
    
    Format the response as a JSON array where each question follows this structure:
    {{
        "question": "Question text here",
        "options": {{
            "A": "Option A text",
            "B": "Option B text", 
            "C": "Option C text",
            "D": "Option D text"
        }},
        "correct_answer": "A",
        "explanation": "Detailed explanation here"
    }}
    
    Make sure the questions are relevant to {topic} and appropriately challenging for aptitude tests."""
    
    try:
        response = llm.invoke(prompt)
        # Try to extract JSON from response
        response_text = response.content
        
        # Find JSON array in response
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            questions = json.loads(json_str)
            return questions
        else:
            st.error("Could not parse questions from AI response")
            return []
            
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

# ---------- Enhanced Quiz System ----------
def start_quiz(questions: List[Dict], timer_minutes: int, topic: str = "Unknown"):
    """Initialize quiz session with enhanced tracking."""
    # Filter out previously used questions
    fresh_questions = filter_unused_questions(questions, len(questions))
    
    # Generate unique quiz ID
    quiz_id = f"quiz_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    st.session_state.quiz_questions = fresh_questions
    st.session_state.quiz_started = True
    st.session_state.current_question = 0
    st.session_state.quiz_answers = {}
    st.session_state.quiz_start_time = time.time()
    st.session_state.timer_setting = timer_minutes * 60  # Convert to seconds
    st.session_state.question_start_time = time.time()
    st.session_state.current_quiz_id = quiz_id
    st.session_state.current_topic = topic
    
    # Mark questions as used
    mark_questions_as_used(fresh_questions)

def display_timer(time_limit: int):
    """Display countdown timer with color coding."""
    if st.session_state.question_start_time is None:
        return time_limit
    
    elapsed = time.time() - st.session_state.question_start_time
    remaining = max(0, time_limit - elapsed)
    
    # Color coding based on remaining time
    if remaining > time_limit * 0.5:
        color = "🟢"
    elif remaining > time_limit * 0.25:
        color = "🟡"
    else:
        color = "🔴"
    
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    
    timer_col, _ = st.columns([1, 3])
    with timer_col:
        st.markdown(f"### {color} Time: {minutes:02d}:{seconds:02d}")
    
    return remaining

def display_quiz_progress():
    """Display quiz progress bar."""
    if st.session_state.quiz_questions:
        progress = (st.session_state.current_question) / len(st.session_state.quiz_questions)
        st.progress(progress)
        st.markdown(f"**Question {st.session_state.current_question + 1} of {len(st.session_state.quiz_questions)}**")

def calculate_results() -> Dict[str, Any]:
    """Calculate quiz results and performance metrics."""
    if not st.session_state.quiz_questions:
        return {}
    
    correct_count = 0
    total_questions = len(st.session_state.quiz_questions)
    question_details = []
    
    for i, question in enumerate(st.session_state.quiz_questions):
        user_answer = st.session_state.quiz_answers.get(i, "Not Answered")
        correct_answer = question["correct_answer"]
        is_correct = user_answer == correct_answer
        
        if is_correct:
            correct_count += 1
        
        question_details.append({
            "question": question["question"],
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "explanation": question["explanation"]
        })
    
    percentage = (correct_count / total_questions) * 100
    total_time = time.time() - st.session_state.quiz_start_time if st.session_state.quiz_start_time else 0
    avg_time_per_question = total_time / total_questions
    
    # Performance category
    if percentage >= 80:
        category = "🏆 Excellent"
    elif percentage >= 60:
        category = "👍 Good"
    elif percentage >= 40:
        category = "📖 Practice Needed"
    else:
        category = "🔄 Keep Studying"
    
    results = {
        "score": f"{correct_count}/{total_questions}",
        "percentage": round(percentage, 1),
        "category": category,
        "total_time": round(total_time, 2),
        "avg_time_per_question": round(avg_time_per_question, 2),
        "question_details": question_details,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Save to history
    save_quiz_to_history(results)
    
    return results

# ---------- Main UI ----------
st.set_page_config(
    page_title="🎯 Enhanced Aptitude Quiz System",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()

st.title("🎯 Enhanced Aptitude Question Generator & Quiz System")
st.markdown("*Advanced RAG-powered aptitude test preparation with quiz history and smart question management*")

# Enhanced sidebar with quiz statistics
with st.sidebar:
    # Quiz statistics
    st.divider()
    st.header("📊 Session Statistics")
    
    stats = get_quiz_statistics()
    if stats:
        st.metric("🎯 Total Quizzes", stats['total_quizzes'])
        st.metric("📝 Questions Answered", stats['total_questions_answered'])
        st.metric("📈 Average Score", f"{stats['average_percentage']}%")
        st.metric("🏆 Best Score", f"{stats['best_score']}%")
        
        if stats['recent_trend']:
            st.write("**Recent Trend:**")
            trend_text = " → ".join([f"{score:.1f}%" for score in stats['recent_trend']])
            st.text(trend_text)
    else:
        st.info("No quiz history yet")
    
    # Question pool status
    st.divider()
    st.header("🔄 Question Pool")
    st.info(f"Used Questions: {len(st.session_state.used_questions)}")
    
    if st.button("🔄 Reset Question Pool"):
        st.session_state.used_questions.clear()
        st.success("✅ Question pool reset!")
        st.rerun()

# Main tabs with enhanced history tab
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Document Upload", "❓ Question Generator", "🏃‍♂️ Quiz System", "📊 Current Results", "📈 Quiz History"])

# Tab 1: Document Upload (unchanged)
with tab1:
    st.header("📁 Document Processing & Management")
    
    # Check for documents in temp folder
    temp_files = load_documents_from_temp_folder()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📂 Documents in Temp Folder")
        
        if temp_files:
            st.success(f"✅ Found {len(temp_files)} document(s) in temp folder")
            
            # Display temp folder files
            file_info = []
            for file_path in temp_files:
                file_name = Path(file_path).name
                file_size = os.path.getsize(file_path) / 1024  # KB
                file_type = file_name.split('.')[-1].upper()
                
                file_info.append({
                    "Filename": file_name,
                    "Type": file_type,
                    "Size": f"{file_size:.1f} KB"
                })
            
            st.dataframe(pd.DataFrame(file_info), use_container_width=True)
            
            # Process temp folder documents
            if st.button("🔄 Process Documents from Temp Folder", type="primary"):
                with st.spinner("Processing documents from temp folder..."):
                    # Extract text from all temp files
                    combined_text = process_temp_folder_files()
                    
                    if combined_text:
                        # Create vector store
                        with st.spinner("Creating embeddings and vector store..."):
                            index, chunks = create_vector_store(combined_text)
                            embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                        
                        if index is not None:
                            st.session_state.vector_store = index
                            st.session_state.document_chunks = chunks
                            st.session_state.embeddings_model = embeddings_model
                            
                            st.success(f"✅ Successfully processed {len(chunks)} text chunks from temp folder")
                            st.info("📚 Documents are ready for question generation!")
                        else:
                            st.error("❌ Failed to create vector store")
                    else:
                        st.error("❌ No text content found in temp folder documents")
        else:
            st.warning("⚠️ No documents found in temp folder")
            st.info("📝 Add PDF, DOCX, or TXT files to the 'temp/' folder in your project directory")
        
        st.divider()
        st.subheader("📤 Alternative: Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload additional files if needed"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
            
            file_info = []
            for file in uploaded_files:
                file_info.append({
                    "Filename": file.name,
                    "Type": file.type,
                    "Size": f"{file.size / 1024:.1f} KB"
                })
            
            st.dataframe(pd.DataFrame(file_info), use_container_width=True)
            
            if st.button("🔄 Process Uploaded Documents"):
                with st.spinner("Processing uploaded documents..."):
                    # Extract text from uploaded files
                    combined_text = process_uploaded_files(uploaded_files)
                    
                    if combined_text:
                        # Create vector store
                        with st.spinner("Creating embeddings and vector store..."):
                            index, chunks = create_vector_store(combined_text)
                            embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                        
                        if index is not None:
                            st.session_state.vector_store = index
                            st.session_state.document_chunks = chunks
                            st.session_state.embeddings_model = embeddings_model
                            
                            st.success(f"✅ Successfully processed {len(chunks)} text chunks")
                            st.info("📚 Documents are ready for question generation!")
                        else:
                            st.error("❌ Failed to create vector store")
                    else:
                        st.error("❌ No text content found in uploaded documents")
    
    with col2:
        st.info("""
        **Project Structure:**
        ```
        your_project/
        ├── app.py
        ├── requirements.txt
        ├── .env
        └── temp/        ← Place files here
            ├── file1.pdf
            ├── file2.docx
            └── file3.txt
        ```
        
        **Supported Formats:**
        - 📄 PDF files
        - 📝 DOCX documents  
        - 📋 TXT files
        
        **Features:**
        - Auto-discovery from temp/
        - Batch processing
        - Multi-format support
        - Error handling
        """)
        
        # Quick stats about temp folder
        if temp_files:
            st.success(f"📊 **Temp folder stats:**\n- {len(temp_files)} files found")
            
            # File type breakdown
            file_types = {}
            for file_path in temp_files:
                ext = Path(file_path).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            
            for ext, count in file_types.items():
                st.write(f"- {count} {ext.upper()} file{'s' if count != 1 else ''}")
        
        # Show current vector store status
        if st.session_state.vector_store is not None:
            st.success("✅ **Vector store ready**")
            chunk_count = len(st.session_state.document_chunks) if hasattr(st.session_state, 'document_chunks') else 0
            st.info(f"📊 {chunk_count} text chunks indexed")
        else:
            st.warning("⏳ **Vector store not ready**")

# Tab 2: Enhanced Question Generator
with tab2:
    st.header("❓ AI-Powered Question Generation")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        topic = st.text_input(
            "📚 Enter Aptitude Topic",
            placeholder="e.g., Logical Reasoning, Quantitative Aptitude, Data Interpretation",
            help="Specify the aptitude topic for question generation"
        )
        
        num_questions = st.slider("🔢 Number of Questions", min_value=1, max_value=30, value=5)
        
        use_rag = st.checkbox("🔍 Use uploaded documents for context", value=True)
        
        # Show question pool status
        unused_count = 0
        if hasattr(st.session_state, 'all_generated_questions'):
            for q in st.session_state.all_generated_questions:
                if generate_question_hash(q) not in st.session_state.used_questions:
                    unused_count += 1
        
        if unused_count > 0:
            st.info(f"💡 {unused_count} unused questions available in the pool")
        
    with col2:
        st.info("""
        **Popular Topics:**
        - Logical Reasoning
        - Quantitative Aptitude
        - Data Interpretation
        - Verbal Ability
        - General Awareness
        
        **Smart Features:**
        - Question deduplication
        - History tracking
        - Pool management
        """)
    
    if st.button("🎯 Generate Questions", type="primary"):
        if not topic.strip():
            st.error("❌ Please enter a topic")
        elif "GROQ_API_KEY" not in os.environ:
            st.error("❌ Groq API key not configured")
        else:
            with st.spinner(f"Generating {num_questions} questions on {topic}..."):
                context = ""
                
                # Use RAG if enabled and documents are available
                if use_rag and st.session_state.vector_store is not None:
                    context = retrieve_context(
                        topic, 
                        st.session_state.vector_store,
                        st.session_state.document_chunks,
                        st.session_state.embeddings_model,
                        top_k=3
                    )
                
                questions = generate_questions(topic, num_questions, context)
                
                if questions:
                    # Add to all generated questions pool
                    st.session_state.all_generated_questions.extend(questions)
                    st.session_state.generated_questions = questions
                    
                    st.success(f"✅ Generated {len(questions)} questions successfully!")
                    
                    # Download option
                    questions_json = json.dumps(questions, indent=2)
                    st.download_button(
                        "📥 Download Questions (JSON)",
                        questions_json,
                        f"questions_{topic.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
                else:
                    st.error("❌ Failed to generate questions")

# Tab 3: Enhanced Quiz System
with tab3:
    st.header("🏃‍♂️ Interactive Quiz System")
    
    if 'generated_questions' not in st.session_state or not st.session_state.generated_questions:
        st.warning("⚠️ Please generate questions first in the Question Generator tab")
    else:
        if not st.session_state.quiz_started:
            # Quiz configuration
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⏰ Timer Settings")
                timer_option = st.selectbox(
                    "Select timer per question:",
                    ["30 seconds (Speed)", "1 minute (Standard)", "2 minutes (Relaxed)", "No timer"]
                )
                
                timer_mapping = {
                    "30 seconds (Speed)": 30,
                    "1 minute (Standard)": 60,
                    "2 minutes (Relaxed)": 120,
                    "No timer": 300  # 5 minutes max
                }
                
                selected_timer = timer_mapping[timer_option]
                
                # Show question availability
                available_questions = filter_unused_questions(
                    st.session_state.generated_questions, 
                    len(st.session_state.generated_questions)
                )
                
                if len(available_questions) < len(st.session_state.generated_questions):
                    st.info(f"🔄 {len(available_questions)} fresh questions available (some questions were used before)")
                else:
                    st.success(f"✨ All {len(available_questions)} questions are fresh!")
            
            with col2:
                st.subheader("📊 Quiz Preview")
                st.info(f"**Questions Available:** {len(available_questions)}")
                st.info(f"**Estimated Time:** {len(available_questions) * (selected_timer/60):.1f} minutes")
                
                # Current topic from session state
                current_topic = getattr(st.session_state, 'current_topic', 'Unknown')
                if hasattr(st.session_state, 'generated_questions') and st.session_state.generated_questions:
                    # Try to infer topic from previous generation
                    st.info(f"**Topic:** {current_topic}")
            
            if st.button("🚀 Start Timed Quiz", type="primary"):
                topic_for_quiz = getattr(st.session_state, 'current_topic', 'Generated Topic')
                start_quiz(st.session_state.generated_questions, selected_timer/60, topic_for_quiz)
                st.rerun()
        
        else:
            # Active quiz interface
            questions = st.session_state.quiz_questions
            current_idx = st.session_state.current_question
            
            if current_idx < len(questions):
                current_q = questions[current_idx]
                
                # Display progress and timer
                display_quiz_progress()
                remaining_time = display_timer(st.session_state.timer_setting)
                
                # Show quiz info
                st.info(f"🎯 Quiz ID: {st.session_state.current_quiz_id} | 📚 Topic: {st.session_state.current_topic}")
                
                # Auto advance when timer expires
                if remaining_time <= 0:
                    if current_idx < len(questions) - 1:
                        st.session_state.current_question += 1
                        st.session_state.question_start_time = time.time()
                        st.rerun()
                    else:
                        # Quiz finished
                        st.session_state.quiz_started = False
                        st.rerun()
                
                st.divider()
                
                # Question display
                st.markdown(f"### Question {current_idx + 1}")
                st.markdown(current_q['question'])
                
                # Answer options
                answer_key = f"q_{current_idx}"
                user_answer = st.radio(
                    "Select your answer:",
                    options=list(current_q['options'].keys()),
                    format_func=lambda x: f"{x}) {current_q['options'][x]}",
                    key=answer_key
                )
                
                # Control buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("⏭️ Next Question"):
                        st.session_state.quiz_answers[current_idx] = user_answer
                        if current_idx < len(questions) - 1:
                            st.session_state.current_question += 1
                            st.session_state.question_start_time = time.time()
                        else:
                            st.session_state.quiz_started = False
                        st.rerun()
                
                with col2:
                    if st.button("🔄 Reset Timer"):
                        st.session_state.question_start_time = time.time()
                        st.rerun()
                
                with col3:
                    if st.button("🏁 Finish Quiz"):
                        st.session_state.quiz_answers[current_idx] = user_answer
                        st.session_state.quiz_started = False
                        st.rerun()
                
                # Auto-refresh for timer update
                time.sleep(1)
                st.rerun()
            
            else:
                # Quiz completed
                st.session_state.quiz_started = False
                st.success("🎉 Quiz Completed!")
                st.rerun()

# Tab 4: Current Results
with tab4:
    st.header("📊 Current Quiz Results")
    
    if st.session_state.quiz_answers and 'quiz_questions' in st.session_state and st.session_state.quiz_questions:
        results = calculate_results()
        
        if results:
            # Performance overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📈 Score", results['score'])
            
            with col2:
                st.metric("🎯 Percentage", f"{results['percentage']}%")
            
            with col3:
                st.metric("⏱️ Total Time", f"{results['total_time']}s")
            
            with col4:
                st.metric("⏰ Avg Time/Q", f"{results['avg_time_per_question']:.1f}s")
            
            # Performance category
            st.markdown(f"## {results['category']}")
            
            # Quiz information
            if hasattr(st.session_state, 'current_quiz_id'):
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🆔 Quiz ID: {st.session_state.current_quiz_id}")
                with col2:
                    st.info(f"📚 Topic: {getattr(st.session_state, 'current_topic', 'Unknown')}")
            
            # Detailed results
            st.subheader("📋 Question-by-Question Review")
            
            for i, detail in enumerate(results['question_details']):
                with st.expander(f"Question {i+1} - {'✅' if detail['is_correct'] else '❌'}"):
                    st.markdown(f"**Question:** {detail['question']}")
                    st.markdown(f"**Your Answer:** {detail['user_answer']}")
                    st.markdown(f"**Correct Answer:** {detail['correct_answer']}")
                    st.markdown(f"**Explanation:** {detail['explanation']}")
            
            # Export results
            results_json = json.dumps(results, indent=2)
            st.download_button(
                "📥 Download Current Results (JSON)",
                results_json,
                f"quiz_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
            
            # Reset quiz option
            if st.button("🔄 Start New Quiz"):
                st.session_state.quiz_answers = {}
                st.session_state.current_question = 0
                st.session_state.quiz_started = False
                st.rerun()
    
    else:
        st.info("📝 Complete a quiz to see results and analytics here")

# Tab 5: Enhanced Quiz History
with tab5:
    st.header("📈 Quiz History & Analytics")
    
    if not st.session_state.quiz_history:
        st.info("📝 No quiz history available yet. Complete some quizzes to see your progress!")
    else:
        # Overall statistics
        stats = get_quiz_statistics()
        
        st.subheader("📊 Overall Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Total Quizzes", stats['total_quizzes'])
        with col2:
            st.metric("📝 Total Questions", stats['total_questions_answered'])
        with col3:
            st.metric("📈 Average Score", f"{stats['average_percentage']}%")
        with col4:
            st.metric("🏆 Best Score", f"{stats['best_score']}%")
        
        # Recent performance trend
        if stats['recent_trend'] and len(stats['recent_trend']) >= 2:
            st.subheader("📈 Recent Trend")
            trend_data = pd.DataFrame({
                'Quiz': [f"Quiz {i+1}" for i in range(len(stats['recent_trend']))],
                'Score (%)': stats['recent_trend']
            })
            st.line_chart(trend_data.set_index('Quiz'))
        
        st.divider()
        
        # Detailed quiz history
        st.subheader("📚 Quiz History Details")
        
        # Create history table
        history_data = []
        for i, quiz in enumerate(reversed(st.session_state.quiz_history)):
            completed_time = datetime.datetime.fromisoformat(quiz['completed_at']).strftime('%Y-%m-%d %H:%M')
            history_data.append({
                'Quiz #': len(st.session_state.quiz_history) - i,
                'Date & Time': completed_time,
                'Topic': quiz.get('topic', 'Unknown'),
                'Score': quiz['results']['score'],
                'Percentage': f"{quiz['results']['percentage']}%",
                'Total Time': f"{quiz['results']['total_time']:.1f}s",
                'Quiz ID': quiz['quiz_id']
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Individual quiz details
        st.subheader("🔍 Detailed Quiz Analysis")
        
        if len(st.session_state.quiz_history) > 0:
            # Quiz selector
            quiz_options = [f"Quiz #{len(st.session_state.quiz_history) - i}: {quiz['topic']} ({datetime.datetime.fromisoformat(quiz['completed_at']).strftime('%Y-%m-%d %H:%M')})" 
                          for i, quiz in enumerate(reversed(st.session_state.quiz_history))]
            
            selected_quiz_idx = st.selectbox("Select quiz to analyze:", range(len(quiz_options)), format_func=lambda x: quiz_options[x])
            
            # Get selected quiz (reverse index since we reversed the display)
            selected_quiz = st.session_state.quiz_history[-(selected_quiz_idx + 1)]
            
            # Display selected quiz details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Score", selected_quiz['results']['score'])
                st.metric("🎯 Percentage", f"{selected_quiz['results']['percentage']}%")
            with col2:
                st.metric("⏱️ Total Time", f"{selected_quiz['results']['total_time']}s")
                st.metric("⏰ Avg Time/Q", f"{selected_quiz['results']['avg_time_per_question']:.1f}s")
            with col3:
                st.metric("📚 Topic", selected_quiz['topic'])
                st.metric("🆔 Quiz ID", selected_quiz['quiz_id'][-8:])  # Show last 8 chars
            
            # Question breakdown
            st.markdown("### 📋 Question Breakdown")
            
            correct_count = 0
            for i, detail in enumerate(selected_quiz['results']['question_details']):
                if detail['is_correct']:
                    correct_count += 1
                
                with st.expander(f"Question {i+1} - {'✅' if detail['is_correct'] else '❌'} {detail['question'][:50]}..."):
                    st.markdown(f"**Full Question:** {detail['question']}")
                    st.markdown(f"**Your Answer:** {detail['user_answer']}")
                    st.markdown(f"**Correct Answer:** {detail['correct_answer']}")
                    st.markdown(f"**Explanation:** {detail['explanation']}")
        
        # Export all history
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export All History"):
                all_history_json = json.dumps(st.session_state.quiz_history, indent=2)
                st.download_button(
                    "Download Complete History (JSON)",
                    all_history_json,
                    f"quiz_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        
        with col2:
            if st.button("🗑️ Clear History", type="secondary"):
                if st.button("⚠️ Confirm Clear History", type="secondary"):
                    st.session_state.quiz_history = []
                    st.session_state.used_questions.clear()
                    st.success("✅ Quiz history cleared!")
                    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
🎯 Enhanced Aptitude Question Generator & Quiz System | Features: Smart Question Management, Quiz History, Performance Analytics<br>
Powered by Groq AI & FAISS Vector Search
</div>
""", unsafe_allow_html=True)