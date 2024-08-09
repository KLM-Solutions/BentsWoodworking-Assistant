import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import time
import os
from dotenv import load_dotenv
import random
import pandas as pd
from docx import Document
from database_operations import get_database_connection, init_db, load_initial_data, get_all_products, query_db_for_keywords, add_product, delete_product, update_product, get_product_by_id
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langsmith import trace
import functools

# Load environment variables
load_dotenv()

# Load API keys and set environment variables
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "Bents-Woodworking-Assistant"

from langsmith import Client
langsmith_client = Client(api_key=LANGCHAIN_API_KEY)

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "bents-woodworking"

# Initialize Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
index = pc.Index(INDEX_NAME)

# YouTube video links
YOUTUBE_LINKS = {
    "Basics of Cabinet Building": "https://www.youtube.com/watch?v=Oeu7ogH2NZU&t=3910s",
    "Graco Ultimate Sprayer": "https://www.youtube.com/watch?v=T8BIpNzdh7M&t=264s",
    "Festool LR32 system": "https://www.youtube.com/watch?v=EO62T1LHdNA"
}

# List of example questions
EXAMPLE_QUESTIONS = [
    "How do TSO Products' Festool accessories improve woodworking precision?",
    "What makes Bits and Bits Company's router bits ideal for woodworking?",
    "What are the benefits of Japanese saws and chisels from Taylor Toolworks?",
    "How does the Festool LR 32 System aid in cabinet making?",
    "What advantages does the Festool Trigger Clamp offer for quick release and one-handed use?",
    "How does the Festool LR 32 Rail ensure precise 32mm hole spacing?",
    "What features of the Festool OF 1400 plunge router are ideal for precision routing?",
    "How does the Festool Vac Sys Head improve clamping in woodworking?",
    "What makes the Festool Midi Vac a top choice for dust extraction?",
    "How does the Festool Bluetooth Switch enhance dust extractor control?"
]

from langsmith import trace

def safe_run_tree(name, run_type):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with trace(name=name, run_type=run_type, client=langsmith_client) as run:
                    result = func(*args, **kwargs)
                    run.end(outputs={"result": str(result)})
                    return result
            except Exception as e:
                st.error(f"Error in LangSmith tracing: {str(e)}")
                return func(*args, **kwargs)
        return wrapper
    return decorator

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text
    import re

def sanitize_id(text):
    # Replace spaces with underscores and remove any other non-alphanumeric characters
    sanitized = re.sub(r'\s+', '_', text)
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
    # Ensure the ID starts with a letter or underscore
    if not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = '_' + sanitized
    # Limit the length of the ID (Pinecone might have a maximum length requirement)
    sanitized = sanitized[:36]  # Adjust this number if needed
    return sanitized

def extract_metadata_from_text(text):
    title = text.split('\n')[0] if text else "Untitled Video"
    return {"title": title}

@safe_run_tree(name="generate_embedding", run_type="llm")
def generate_embedding(text):
    embeddings = OpenAIEmbeddings()
    with get_openai_callback() as cb:
        embedding = embeddings.embed_query(text)
    return embedding

def upsert_transcript(transcript_text, metadata):
    chunks = [transcript_text[i:i+8000] for i in range(0, len(transcript_text), 8000)]
    for i, chunk in enumerate(chunks):
        try:
            embedding = generate_embedding(chunk)
            if embedding:
                chunk_metadata = metadata.copy()
                chunk_metadata['text'] = chunk
                sanitized_title = sanitize_id(metadata['title'])
                chunk_id = f"{sanitized_title}_chunk_{i}"
                chunk_metadata['chunk_id'] = chunk_id
                index.upsert([(chunk_id, embedding, chunk_metadata)])
            else:
                st.warning(f"Failed to generate embedding for chunk {i} of {metadata['title']}")
        except Exception as e:
            st.error(f"Error upserting chunk {i} of {metadata['title']}: {str(e)}")

def query_pinecone(query, index):
    query_embedding = generate_embedding(query)
    if query_embedding:
        result = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        return [(match['metadata']['title'], match['metadata']['text']) for match in result['matches']]
    else:
        return []

@safe_run_tree(name="generate_keywords", run_type="llm")
def generate_keywords(text):
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    system_message = SystemMessage(content="You are a specialized keyword extraction system for woodworking terminology. Your task is to identify and extract the most relevant technical terms, tool names, materials, techniques, and concepts from the given text. Adhere to these guidelines:\n\n1. Focus exclusively on woodworking-related terms and concepts.\n2. Prioritize specificity over generality in your selections.\n3. Include both common and specialized woodworking terminology.\n4. If brand names are mentioned, include them only if they're standard in the industry.\n5. Aim for a mix of nouns (tools, materials) and verb phrases (techniques, processes).\n6. If the text is long, focus on the most significant terms.\n9. Ensure each keyword or phrase is distinct and non-redundant and Separate keywords with commas.")
    human_message = HumanMessage(content=f"Generate 3-5 highly relevant and specific keywords or short phrases from this text, separated by commas. Focus on technical terms, tool names, or specific woodworking techniques: {text}")
    
    with get_openai_callback() as cb:
        response = chat([system_message, human_message])
    
    keywords = response.content.strip().split(',')
    return [keyword.strip().lower() for keyword in keywords if keyword.strip()]

@safe_run_tree(name="get_answer", run_type="chain")
def get_answer(context, user_query):
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    system_message = SystemMessage(content="You are a specialized keyword extraction system for woodworking queries. Your task is to identify and extract the most relevant terms from the user's question. Follow these guidelines:\n\n1. Focus on woodworking-specific terminology, techniques, tools, and concepts.\n2. Prioritize technical terms and specific woodworking.\n3. Include tool names, wood types, joinery methods, and finishing techniques if mentioned.\n4. Extract any measurement terms or numerical values related to woodworking.\n5. If the query mentions specific brands or products, include them.\n6. Look for action verbs related to woodworking processes.\n7. Consider any terms related to wood properties or characteristics.\n8. Aim for 3-5 highly relevant keywords or short phrases.\n9. Separate keywords with commas and avoid redundancy.\n10. Do not add any explanations or commentary - output only the keywords.")
    human_message = HumanMessage(content=f"Answer the following question based on the context: {context}\n\nQuestion: {user_query}")
    
    with get_openai_callback() as cb:
        response = chat([system_message, human_message])
    initial_answer = response.content
    
    query_keywords = generate_keywords(user_query)
    answer_keywords = generate_keywords(initial_answer)
    all_keywords = list(set(query_keywords + answer_keywords))
    related_products = query_db_for_keywords(all_keywords)
    
    system_message_2 = SystemMessage(content="You are Jason Bent's woodworking expertise embodied in an AI. Your task:1. Compare query and answer keywords, focusing on woodworking terms.2. Search the context for relevant information using these keywords.3. Integrate useful aspects of related products without naming them.4. Craft a comprehensive response that: Directly addresses the user's query, Incorporates relevant information from the initial answer and context, Reflects Jason's expertise and communication style, Includes specific techniques, tips, or advice, Balances technical accuracy with accessibility, 5. Reference video titles and timestamps for key information., 6. Ensure the answer is cohesive, relevant, and follows safety best practices.")
    human_message_2 = HumanMessage(content=f"Initial Answer: {initial_answer}\n\nRelated Products: {related_products}\n\nPlease provide a final answer that incorporates information from the related products, if relevant, without mentioning specific product names or including any links.")
    
    with get_openai_callback() as cb:
        final_response = chat([system_message_2, human_message_2])
    final_answer = final_response.content
    
    return final_answer, related_products, all_keywords

def process_query(query):
    if query:
        with st.spinner("Searching for the best answer..."):
            matches = query_pinecone(query, index)
            if matches:
                retrieved_texts = [text for _, text in matches]
                retrieved_titles = [title for title, _ in matches]
                context = " ".join([f"Title: {title}\n{text}" for title, text in zip(retrieved_titles, retrieved_texts)])
                final_answer, related_products, keywords = get_answer(context, query)
                
                # Find the related video
                related_video = None
                for title in retrieved_titles:
                    if title in YOUTUBE_LINKS:
                        related_video = title
                        break

                # Display the answer
                st.write(final_answer)
                
                # Add some vertical space
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                # Create two columns: video and related products
                col1, space, col2 = st.columns([0.47, 0.20, 0.47])
                
                # Column 1: Display YouTube video
                with col1:
                    st.subheader("Related Video")
                    video_displayed = False
                    for title in retrieved_titles:
                        if title in YOUTUBE_LINKS:
                            video_id = YOUTUBE_LINKS[title].split("v=")[1].split("&")[0]
                            st.markdown(f"""
                                <iframe width="100%" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
                            """, unsafe_allow_html=True)
                            st.caption(f"Video: {title}")
                            video_displayed = True
                            break  # Display only the first matching video
                    if not video_displayed:
                        st.write("No related video found.")
                # Middle column: Empty space
                with space:
                    st.empty()
                
                # Column 2: Display related products
                with col2:
                    st.subheader("Related Products")
                    if related_products:
                        for product in related_products:
                            st.markdown(f"[{product[1]}]({product[3]})")
                    else:
                        st.write("No related products found.")
                
                # Add some more vertical space after the columns
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append((query, final_answer, related_products, related_video))
            else:
                st.warning("I couldn't find a specific answer to your question. Please try rephrasing or ask something else.")
    else:
        st.warning("Please enter a question before searching.")

def query_interface():
    st.header("Woodworking Assistant")

    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = random.sample(EXAMPLE_QUESTIONS, 3)
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

    st.subheader("Popular Questions")
    for question in st.session_state.selected_questions:
        if st.button(question, key=question):
            st.session_state.current_query = question

    user_query = st.text_input("What would you like to know about woodworking?")
    if st.button("Get Answer"):
        st.session_state.current_query = user_query

    if st.session_state.current_query:
        st.subheader("Response:")
        process_query(st.session_state.current_query)
        st.session_state.current_query = ""

    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.header("Recent Questions and Answers")
        for i, (q, a, products, video_title) in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {q}"):
                st.write(f"A: {a}")
                
                # Create two columns for video and products
                col1, col2 = st.columns([0.5, 0.5])
                
                # Column 1: Display related video
                with col1:
                    st.subheader("Related Video")
                    if video_title and video_title in YOUTUBE_LINKS:
                        video_id = YOUTUBE_LINKS[video_title].split("v=")[1].split("&")[0]
                        st.markdown(f"""
                            <iframe width="100%" height="215" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
                        """, unsafe_allow_html=True)
                        st.caption(f"Video: {video_title}")
                    else:
                        st.write("No related video found.")
                
                # Column 2: Display related products
                with col2:
                    st.subheader("Related Products")
                    if products:
                        for product in products:
                            st.markdown(f"[{product[1]}]({product[3]})")
                    else:
                        st.write("No related products found.")

def database_interface():
    

    st.subheader("All Products")
    products = get_all_products()
    if products:
        df = pd.DataFrame(products, columns=['ID', 'Title', 'Tags', 'Links'])
        st.dataframe(df)
    else:
        st.write("The database is empty.")

    st.subheader("Add New Product")
    new_title = st.text_input("Title")
    new_tags = st.text_input("Tags (comma-separated)")
    new_link = st.text_input("Link")
    if st.button("Add Product"):
        if new_title and new_tags and new_link:
            add_product(new_title, new_tags, new_link)
            st.success("Product added successfully!")
        else:
            st.warning("Please fill in all fields.")

    st.subheader("Update Product")
    update_id = st.number_input("Enter Product ID to update", min_value=1, step=1)
    product = get_product_by_id(update_id)
    if product:
        update_title = st.text_input("New Title", value=product[1])
        update_tags = st.text_input("New Tags", value=product[2])
        update_link = st.text_input("New Link", value=product[3])
        if st.button("Update Product"):
            update_product(update_id, update_title, update_tags, update_link)
            st.success(f"Product with ID {update_id} updated successfully!")
    else:
        st.warning(f"No product found with ID {update_id}")

    st.subheader("Delete Product")
    delete_id = st.number_input("Enter Product ID to delete", min_value=1, step=1)
    if st.button("Delete Product"):
        delete_product(delete_id)
        st.success(f"Product with ID {delete_id} deleted successfully!")

def main():
    st.set_page_config(page_title="Bent's Woodworking Assistant", layout="wide")

    if not LANGCHAIN_API_KEY:
        st.warning("LangSmith API key is not set. Some features may not work properly.")

    # Add the logo to the main page
    st.image("bents logo.png", width=150)
    st.title("Bent's Woodworking Assistant")

    # Initialize database connection
    conn = get_database_connection()
    init_db(conn)
    load_initial_data()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Query Interface", "Database Management"])

    if page == "Query Interface":
        query_interface()
    elif page == "Database Management":
        database_interface()

    # Sidebar for file upload and metadata (only in Query Interface)
    if page == "Query Interface":
        with st.sidebar:
            st.header("Upload Transcripts")
            uploaded_files = st.file_uploader("Upload YouTube Video Transcripts (DOCX)", type="docx", accept_multiple_files=True)
            if uploaded_files:
                all_metadata = []
                total_token_count = 0
                for uploaded_file in uploaded_files:
                    transcript_text = extract_text_from_docx(uploaded_file)
                    metadata = extract_metadata_from_text(transcript_text)
                    all_metadata.append((metadata, transcript_text))
                st.subheader("Uploaded Transcripts")
                for metadata, _ in all_metadata:
                    st.text(f"Title: {metadata['title']}")
                if st.button("Upsert All Transcripts"):
                    with st.spinner("Upserting transcripts..."):
                        for metadata, transcript_text in all_metadata:
                            upsert_transcript(transcript_text, metadata)
                        st.success("All transcripts upserted successfully!")

if __name__ == "__main__":
    main()
