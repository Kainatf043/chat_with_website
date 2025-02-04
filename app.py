import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import google.generativeai as genai
import os


st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.header("🤖 Chat with Websites")
# ✅ Load environment variables
load_dotenv()

# ✅ Sidebar: Google API key input
with st.sidebar.expander("🔒 Google API", expanded=False):
    st.markdown('[Click here](https://console.cloud.google.com/apis/credentials) to get your Google API key')
    google_api_key = st.text_input("🔑 Enter your API Key", type="password")

# ✅ Configure API Key only if it's provided
if google_api_key and google_api_key.strip():
    genai.configure(api_key=google_api_key)
else:
    st.warning("⚠️ Please enter a valid Google API Key in the sidebar.")
    st.stop()  # Prevent further execution if API key is missing

# ✅ Function to load and vectorize website content
def get_vectorstore_from_url(url):
    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        document_chunks = text_splitter.split_documents(document)
        document_chunks = [chunk.page_content for chunk in document_chunks if chunk.page_content]
        
        if not document_chunks:
            st.error("No valid text extracted from the website.")
            return None

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(document_chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"❌ Error loading website content: {str(e)}")
        return None

# ✅ Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question based only on the provided context. If the answer is not available, respond with:
    'The answer is not available in the context.'
    
    Context: {context}
    Question: {question}
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# ✅ Function to get response from vector store
def get_response(user_input, vector_store):
    if not vector_store:
        return "Vector store is not initialized. Please process a website first."

    docs = vector_store.similarity_search(user_input)
    if not docs:
        return "No relevant information found in the website content."

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_input}, return_only_outputs=True)
    return response["output_text"]

# ✅ Streamlit UI
st.title("🤖 Chat with Websites using Gemini AI")

# ✅ Sidebar for website URL input
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("🌐 Enter Website URL")
    
    if st.button("🔍 Process Website"):
        if website_url:
            with st.spinner("Processing website... ⏳"):
                vector_store = get_vectorstore_from_url(website_url)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.vector_store_created = True
                    st.success("✅ Website content processed successfully!")
        else:
            st.error("❌ Please enter a valid URL.")

# ✅ Check if vector store is created before allowing chat
if "vector_store_created" in st.session_state and st.session_state.vector_store_created:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I am a bot. How can I assist you?")
        ]

    user_query = st.chat_input("💬 Ask a question based on the website content...")

    if user_query:
        response = get_response(user_query, st.session_state.vector_store)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
