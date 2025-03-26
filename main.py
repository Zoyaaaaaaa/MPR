# import os
# from dotenv import load_dotenv
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain.prompts import PromptTemplate
# from langchain.vectorstores import Neo4jVector
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.graphs import Neo4jGraph
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain.chains.graph_qa.cypher import GraphCypherQAChain
# import streamlit as st
# import tempfile
# from neo4j import GraphDatabase

# def main():
#     st.set_page_config(
#         layout="wide",
#         page_title="Graphy v1",
#         page_icon=":graph:"
#     )
#     st.sidebar.image('logo.png', use_column_width=True) 
#     with st.sidebar.expander("Expand Me"):
#         st.markdown("""
#     This application allows you to upload a PDF file, extract its content into a Neo4j graph database, and perform queries using natural language.
#     It leverages LangChain and OpenAI's GPT models to generate Cypher queries that interact with the Neo4j database in real-time.
#     """)
#     st.title("Graphy: Realtime GraphRAG App")

#     load_dotenv()

#     # Set OpenAI API key
#     if 'OPENAI_API_KEY' not in st.session_state:
#         st.sidebar.subheader("OpenAI API Key")
#         openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type='password')
#         if openai_api_key:
#             os.environ['OPENAI_API_KEY'] = openai_api_key
#             st.session_state['OPENAI_API_KEY'] = openai_api_key
#             st.sidebar.success("OpenAI API Key set successfully.")
#             embeddings = OpenAIEmbeddings()
#             llm = ChatOpenAI(model_name="gpt-4o")  # Use model that supports function calling
#             st.session_state['embeddings'] = embeddings
#             st.session_state['llm'] = llm
#     else:
#         embeddings = st.session_state['embeddings']
#         llm = st.session_state['llm']

#     # Initialize variables
#     neo4j_url = None
#     neo4j_username = None
#     neo4j_password = None
#     graph = None

#     # Set Neo4j connection details
#     if 'neo4j_connected' not in st.session_state:
#         st.sidebar.subheader("Connect to Neo4j Database")
#         neo4j_url = st.sidebar.text_input("Neo4j URL:", value="neo4j+s://<your-neo4j-url>")
#         neo4j_username = st.sidebar.text_input("Neo4j Username:", value="neo4j")
#         neo4j_password = st.sidebar.text_input("Neo4j Password:", type='password')
#         connect_button = st.sidebar.button("Connect")
#         if connect_button and neo4j_password:
#             try:
#                 graph = Neo4jGraph(
#                     url=neo4j_url, 
#                     username=neo4j_username, 
#                     password=neo4j_password
#                 )
#                 st.session_state['graph'] = graph
#                 st.session_state['neo4j_connected'] = True
#                 # Store connection parameters for later use
#                 st.session_state['neo4j_url'] = neo4j_url
#                 st.session_state['neo4j_username'] = neo4j_username
#                 st.session_state['neo4j_password'] = neo4j_password
#                 st.sidebar.success("Connected to Neo4j database.")
#             except Exception as e:
#                 st.error(f"Failed to connect to Neo4j: {e}")
#     else:
#         graph = st.session_state['graph']
#         neo4j_url = st.session_state['neo4j_url']
#         neo4j_username = st.session_state['neo4j_username']
#         neo4j_password = st.session_state['neo4j_password']

#     # Ensure that the Neo4j connection is established before proceeding
#     if graph is not None:
#         # File uploader
#         uploaded_file = st.file_uploader("Please select a PDF file.", type="pdf")

#         if uploaded_file is not None and 'qa' not in st.session_state:
#             with st.spinner("Processing the PDF..."):
#                 # Save uploaded file to temporary file
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#                     tmp_file.write(uploaded_file.read())
#                     tmp_file_path = tmp_file.name

#                 # Load and split the PDF
#                 loader = PyPDFLoader(tmp_file_path)
#                 pages = loader.load_and_split()

#                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
#                 docs = text_splitter.split_documents(pages)

#                 lc_docs = []
#                 for doc in docs:
#                     lc_docs.append(Document(page_content=doc.page_content.replace("\n", ""), 
#                     metadata={'source': uploaded_file.name}))

#                 # Clear the graph database
#                 cypher = """
#                   MATCH (n)
#                   DETACH DELETE n;
#                 """
#                 graph.query(cypher)

#                 # Define allowed nodes and relationships
#                 allowed_nodes = ["Patient", "Disease", "Medication", "Test", "Symptom", "Doctor"]
#                 allowed_relationships = ["HAS_DISEASE", "TAKES_MEDICATION", "UNDERWENT_TEST", "HAS_SYMPTOM", "TREATED_BY"]

#                 # Transform documents into graph documents
#                 transformer = LLMGraphTransformer(
#                     llm=llm,
#                     allowed_nodes=allowed_nodes,
#                     allowed_relationships=allowed_relationships,
#                     node_properties=False, 
#                     relationship_properties=False
#                 ) 

#                 graph_documents = transformer.convert_to_graph_documents(lc_docs)
#                 graph.add_graph_documents(graph_documents, include_source=True)

#                 # Use the stored connection parameters
#                 index = Neo4jVector.from_existing_graph(
#                     embedding=embeddings,
#                     url=neo4j_url,
#                     username=neo4j_username,
#                     password=neo4j_password,
#                     database="neo4j",
#                     node_label="Patient",  # Adjust node_label as needed
#                     text_node_properties=["id", "text"], 
#                     embedding_node_property="embedding", 
#                     index_name="vector_index", 
#                     keyword_index_name="entity_index", 
#                     search_type="hybrid" 
#                 )

#                 st.success(f"{uploaded_file.name} preparation is complete.")

#                 # Retrieve the graph schema
#                 schema = graph.get_schema

#                 # Set up the QA chain
#                 template = """
#                 Task: Generate a Cypher statement to query the graph database.

#                 Instructions:
#                 Use only relationship types and properties provided in schema.
#                 Do not use other relationship types or properties that are not provided.

#                 schema:
#                 {schema}

#                 Note: Do not include explanations or apologies in your answers.
#                 Do not answer questions that ask anything other than creating Cypher statements.
#                 Do not include any text other than generated Cypher statements.

#                 Question: {question}""" 

#                 question_prompt = PromptTemplate(
#                     template=template, 
#                     input_variables=["schema", "question"] 
#                 )

#                 qa = GraphCypherQAChain.from_llm(
#                     llm=llm,
#                     graph=graph,
#                     cypher_prompt=question_prompt,
#                     verbose=True,
#                     allow_dangerous_requests=True
#                 )
#                 st.session_state['qa'] = qa
#     else:
#         st.warning("Please connect to the Neo4j database before you can upload a PDF.")

#     if 'qa' in st.session_state:
#         st.subheader("Ask a Question")
#         with st.form(key='question_form'):
#             question = st.text_input("Enter your question:")
#             submit_button = st.form_submit_button(label='Submit')

#         if submit_button and question:
#             with st.spinner("Generating answer..."):
#                 res = st.session_state['qa'].invoke({"query": question})
#                 st.write("\n**Answer:**\n" + res['result'])

# if __name__ == "__main__":
#     main()

import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
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
            st.session_state['llm'] = ChatOpenAI(model_name="gpt-4", temperature=0)
            st.sidebar.success("OpenAI API Key set successfully.")
            return True
        return False
    return True

def initialize_neo4j():
    """Initialize Neo4j connection and store in session state"""
    if 'graph' not in st.session_state:
        try:
            st.session_state['graph'] = Neo4jGraph(
                url="neo4j+s://3b7186ae.databases.neo4j.io",
                username="neo4j",
                password="RkMSaWhFwHQdUpuhJqCaCLnjJx9peKE23EpHIdsKxqM"
            )
            # Test connection
            st.session_state['graph'].query("RETURN 1 AS test")
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
        # Step 1: Save uploaded file
        status_text.text("Saving uploaded file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        progress_bar.progress(0.05)

        # Step 2: Load and split PDF
        status_text.text("Loading PDF document...")
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        progress_bar.progress(0.20)

        status_text.text("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)
        progress_bar.progress(0.30)

        lc_docs = [Document(
            page_content=doc.page_content.replace("\n", " "), 
            metadata={'source': uploaded_file.name}
        ) for doc in docs]

        # Step 3: Clear existing graph (if this is a new document)
        if 'current_document' not in st.session_state:
            status_text.text("Preparing database...")
            st.session_state['graph'].query("MATCH (n) DETACH DELETE n;")
            st.session_state['current_document'] = uploaded_file.name
        progress_bar.progress(0.40)

        # Step 4: Define legal domain schema
        allowed_nodes = ["Case", "Law", "Court", "Lawyer", "Suspect", "Victim", "Judge", "LegalPrecedent"]
        allowed_relationships = [
            "IS_ACCUSED_OF", "DEFENDED", "REFERENCES", 
            "HEARD_BY", "INVOLVES", "SET_PRECEDENT", 
            "APPEALED_TO", "PROSECUTED_BY"
        ]
        progress_bar.progress(0.45)

        # Step 5: Transform documents to graph
        status_text.text("Analyzing legal content...")
        transformer = LLMGraphTransformer(
            llm=st.session_state['llm'],
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            node_properties=False,
            relationship_properties=False
        )

        # Process in batches
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
                st.warning(f"Skipped batch {current_batch} due to error: {str(e)}")
                continue
                
            progress = 0.45 + (0.40 * (i + batch_size) / len(lc_docs))
            progress_bar.progress(min(0.85, progress))
            time.sleep(1)  # Rate limiting

        # Step 6: Add to graph database
        status_text.text("Building knowledge graph...")
        st.session_state['graph'].add_graph_documents(graph_documents, include_source=True)
        progress_bar.progress(0.95)

        # Step 7: Create/update vector index
        status_text.text("Updating search index...")
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
        st.success(f"Document '{uploaded_file.name}' processed successfully!")
        return True
        
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
    """Set up the QA chain with few-shot examples for legal queries"""
    examples = [
        {
            "question": "Which cases referenced the Indian Penal Code Section 302?",
            "query": "MATCH (c:Case)-[:REFERENCES]->(l:Law {section: 'IPC 302'}) RETURN c.title, c.date, l.description"
        },
        {
            "question": "Who defended the suspects in the XYZ corruption case?",
            "query": "MATCH (c:Case {title: 'XYZ corruption case'})<-[:IS_ACCUSED_OF]-(s:Suspect)<-[:DEFENDED]-(l:Lawyer) RETURN l.name, l.bar_id"
        },
        {
            "question": "What precedents were set by cases involving murder charges?",
            "query": "MATCH (c:Case)-[:INVOLVES]->(:Law {section: 'IPC 302'}), (c)-[:SET_PRECEDENT]->(p:LegalPrecedent) RETURN p.case_id, p.law_applied, p.outcome"
        },
        {
            "question": "Which courts heard appeals related to cyber crime laws?",
            "query": "MATCH (c:Case)-[:REFERENCES]->(:Law {section: 'IT Act'}), (c)-[:APPEALED_TO]->(court:Court) RETURN DISTINCT court.name, court.location"
        }
    ]

    example_prompt = PromptTemplate.from_template(
        "Question: {question}\nCypher Query: {query}"
    )

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""You are a legal expert creating Cypher queries for a case law database.
        Use only these node types and relationships:
        
        Nodes:
        - Case (title, date, status)
        - Law (section, description)
        - Court (name, location)
        - Lawyer (name, bar_id)
        - Suspect (name)
        - Victim (name)
        - LegalPrecedent (case_id, outcome)
        
        Relationships:
        - (Suspect)-[:IS_ACCUSED_OF]->(Case)
        - (Lawyer)-[:DEFENDED]->(Suspect)
        - (Case)-[:REFERENCES]->(Law)
        - (Case)-[:HEARD_BY]->(Court)
        - (Case)-[:INVOLVES]->(Victim)
        - (Case)-[:SET_PRECEDENT]->(LegalPrecedent)
        - (Case)-[:APPEALED_TO]->(Court)
        
        Current schema:
        {schema}
        
        Example queries:""",
        suffix="Question: {question}\nCypher Query:",
        input_variables=["schema", "question"]
    )

    st.session_state['qa'] = GraphCypherQAChain.from_llm(
        llm=st.session_state['llm'],
        graph=st.session_state['graph'],
        cypher_prompt=prompt,
        verbose=True,
        validate_cypher=True,
        allow_dangerous_requests=True
    )

    # Initialize retriever for fallback
    st.session_state['retriever'] = Neo4jVector.from_existing_graph(
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

def handle_query(question):
    """Handle user queries with Cypher and fallback to retrieval"""
    try:
        # First try with Cypher query
        result = st.session_state['qa'].invoke({"query": question})
        
        # Fallback to vector search if needed
        if not result['result'] or "I don't know" in result['result']:
            similar_docs = st.session_state['retriever'].similarity_search(question, k=3)
            if similar_docs:
                context = "\n\n".join([doc.page_content for doc in similar_docs])
                response = st.session_state['llm'].invoke(
                    f"Based on these legal excerpts:\n{context}\n\nQuestion: {question}\nAnswer:"
                )
                return response.content
            return "I couldn't find relevant information in the documents."
        
        return result['result']
    
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return "Error processing your query. Please try again."

def main():
    st.set_page_config(
        layout="wide",
        page_title="Legal GraphRAG",
        page_icon=":balance_scale:"
    )
    st.sidebar.image('logo.png', use_column_width=True)
    with st.sidebar.expander("About This App"):
        st.markdown("""
        This application analyzes legal documents, extracts entities and relationships into a knowledge graph,
        and answers questions using both structured queries and document retrieval.
        """)
    st.title("Legal Case Knowledge Graph")

    load_dotenv()

    # Initialize components
    if not initialize_openai():
        st.warning("Please enter your OpenAI API key to continue.")
        return
        
    if not initialize_neo4j():
        return

    # Document processing
    uploaded_file = st.file_uploader("Upload legal document (PDF)", type="pdf")
    if uploaded_file and ('current_document' not in st.session_state or 
                         st.session_state['current_document'] != uploaded_file.name):
        if process_document(uploaded_file):
            setup_qa_chain()

    # Query interface
    if 'qa' in st.session_state:
        st.subheader("Ask About Cases")
        with st.form(key='query_form'):
            question = st.text_area("Enter your legal question:", height=100)
            submit = st.form_submit_button("Submit Query")
            
            if submit and question:
                with st.spinner("Analyzing legal records..."):
                    answer = handle_query(question)
                    
                    st.markdown(f"**Answer:**\n\n{answer}")
                    
                    if hasattr(st.session_state['qa'], 'last_cypher_query'):
                        with st.expander("View generated Cypher query"):
                            st.code(st.session_state['qa'].last_cypher_query)

if __name__ == "__main__":
    main()
