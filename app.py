import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Neo4jVector
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
import streamlit as st
import tempfile
import time

def initialize_openai():
    """Initialize OpenAI components and store in session state"""
    if 'OPENAI_API_KEY' not in st.session_state:
        st.sidebar.subheader("OpenAI API Key")
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type='password')
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
            st.session_state['OPENAI_API_KEY'] = openai_api_key
            st.session_state['embeddings'] = OpenAIEmbeddings()
            st.session_state['llm'] = ChatOpenAI(model_name="gpt-4", temperature=0)  # Changed to more reliable model
            st.sidebar.success("OpenAI API Key set successfully.")
            return True
        return False
    return True

def initialize_neo4j():
    """Initialize Neo4j connection and store in session state"""
    if 'graph' not in st.session_state:
        try:
            # Test connection first
            test_graph = Neo4jGraph(
                url="neo4j+s://3b7186ae.databases.neo4j.io",
                username="neo4j",
                password="RkMSaWhFwHQdUpuhJqCaCLnjJx9peKE23EpHIdsKxqM"
            )
            test_graph.query("RETURN 1 AS test")  # Simple test query
            st.session_state['graph'] = test_graph
            st.sidebar.success("Connected to Neo4j database.")
            return True
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {e}")
            return False
    return True

def process_document(uploaded_file):
    """Process uploaded PDF and create graph structure"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Save the uploaded file (5% progress)
        status_text.text("Saving uploaded file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        progress_bar.progress(0.05)
        time.sleep(0.5)

        # Step 2: Load and split the PDF (20% progress)
        status_text.text("Loading PDF document...")
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        progress_bar.progress(0.20)
        time.sleep(0.5)

        status_text.text("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)
        progress_bar.progress(0.30)
        time.sleep(0.5)

        lc_docs = [Document(
            page_content=doc.page_content.replace("\n", " "), 
            metadata={'source': uploaded_file.name}
        ) for doc in docs]

        # Step 3: Clear the graph database (40% progress)
        status_text.text("Preparing database...")
        st.session_state['graph'].query("MATCH (n) DETACH DELETE n;")
        progress_bar.progress(0.40)
        time.sleep(0.5)

        # Step 4: Define legal domain schema (45% progress)
        status_text.text("Setting up legal schema...")
        allowed_nodes = ["Case", "Law", "Court", "Lawyer", "Suspect", "Victim", "Judge", "LegalPrecedent"]
        allowed_relationships = [
            "IS_ACCUSED_OF", "DEFENDED", "REFERENCES", 
            "HEARD_BY", "INVOLVES", "SET_PRECEDENT", 
            "APPEALED_TO", "PROSECUTED_BY"
        ]
        progress_bar.progress(0.45)
        time.sleep(0.5)

        # Step 5: Transform documents to graph (45-85% progress)
        status_text.text("Analyzing legal content...")
        transformer = LLMGraphTransformer(
            llm=st.session_state['llm'],
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            node_properties=False,
            relationship_properties=False
        ) 

        batch_size = 3
        total_batches = (len(lc_docs) // batch_size) + 1
        graph_documents = []
        
        for i in range(0, len(lc_docs), batch_size):
            batch = lc_docs[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            status_text.text(f"Processing batch {current_batch}/{total_batches}...")
            
            try:
                batch_graph = transformer.convert_to_graph_documents(batch)
                graph_documents.extend(batch_graph)
            except Exception as e:
                st.warning(f"Warning: Issue processing batch {current_batch}: {str(e)}")
                continue
                
            # Calculate progress (45% + up to 40% for processing)
            progress = 0.45 + (0.40 * (i + batch_size) / len(lc_docs))
            progress_bar.progress(min(0.85, progress))
            time.sleep(1)  # Add delay to avoid rate limiting

        # Step 6: Add to graph database (85-95% progress)
        status_text.text("Building knowledge graph...")
        try:
            st.session_state['graph'].add_graph_documents(graph_documents, include_source=True)
            progress_bar.progress(0.95)
        except Exception as e:
            st.error(f"Failed to add documents to graph: {str(e)}")
            return False
        time.sleep(0.5)

        # Step 7: Create vector index (95-100% progress)
        status_text.text("Creating search index...")
        try:
            Neo4jVector.from_existing_graph(
                embedding=st.session_state['embeddings'],
                url="neo4j+s://3b7186ae.databases.neo4j.io",
                username="neo4j",
                password="RkMSaWhFwHQdUpuhJqCaCLnjJx9peKE23EpHIdsKxqM",
                database="neo4j",
                node_label="Case",
                text_node_properties=["title", "description"],
                embedding_node_property="embedding",
                index_name="legal_vector_index"
            )
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            time.sleep(1)
            st.success(f"Legal document '{uploaded_file.name}' processed successfully.")
            return True
        except Exception as e:
            st.error(f"Failed to create vector index: {str(e)}")
            return False
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False
    finally:
        try:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
        except:
            pass

def setup_qa_chain():
    """Set up the QA chain with legal-specific prompt"""
    template = """
    You are a legal expert creating Cypher queries for a case law database.
    Use only the following node types and relationships:
    
    Nodes:
    - Case (id, title, date, court, type, status)
    - Law (section, description, jurisdiction)
    - Court (name, location, type)
    - Lawyer (name, bar_id, specialization)
    - Suspect (name, age, criminal_record)
    - Victim (name, age, status)
    - Judge (name, experience)
    - LegalPrecedent (case_id, law_applied, outcome)
    
    Relationships:
    - (Suspect)-[:IS_ACCUSED_OF]->(Case)
    - (Lawyer)-[:DEFENDED]->(Suspect)
    - (Case)-[:REFERENCES]->(Law)
    - (Case)-[:HEARD_BY]->(Court)
    - (Case)-[:INVOLVES]->(Victim)
    - (Case)-[:SET_PRECEDENT]->(LegalPrecedent)
    - (Case)-[:APPEALED_TO]->(Court)
    - (Prosecutor)-[:PROSECUTED_BY]->(Case)
    
    Schema:
    {schema}
    
    Question: {question}
    
    Return only the Cypher query with no additional explanation or text.
    """

    question_prompt = PromptTemplate(
        template=template, 
        input_variables=["schema", "question"] 
    )

    st.session_state['qa'] = GraphCypherQAChain.from_llm(
        llm=st.session_state['llm'],
        graph=st.session_state['graph'],
        cypher_prompt=question_prompt,
        verbose=True,
        allow_dangerous_requests=True
    )


# ... [rest of your code remains the same]
def main():
    st.set_page_config(
        layout="wide",
        page_title="Legal GraphRAG",
        page_icon=":balance_scale:"
    )
    st.sidebar.image('logo.png', use_column_width=True) 
    with st.sidebar.expander("About This App"):
        st.markdown("""
        This application allows you to upload legal PDF documents, extract their content into a Neo4j graph database, 
        and perform queries using natural language. It's specialized for legal case analysis.
        """)
    st.title("Legal Case Graph Analyzer")

    load_dotenv()

    # Initialize components
    if not initialize_openai():
        st.warning("Please enter your OpenAI API key to continue.")
        return
        
    if not initialize_neo4j():
        return

    # File uploader
    uploaded_file = st.file_uploader("Upload a legal document (PDF)", type="pdf")

    if uploaded_file is not None and 'qa' not in st.session_state:
        if process_document(uploaded_file):
            setup_qa_chain()

    if 'qa' in st.session_state:
        st.subheader("Query the Legal Database")
        with st.form(key='question_form'):
            question = st.text_input("Ask a question about the case:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and question:
            with st.spinner("Researching legal records..."):
                try:
                    res = st.session_state['qa'].invoke({"query": question})
                    st.markdown(f"**Legal Analysis:**\n\n{res['result']}")
                except Exception as e:
                    st.error(f"Error processing your query: {e}")

if __name__ == "__main__":
    main()
