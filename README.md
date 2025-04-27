# Breast Cancer Knowledge Explorer

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-orange)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-brightgreen)

A sophisticated clinical knowledge graph system for exploring breast cancer data using natural language queries. This tool helps healthcare professionals and researchers analyze relationships between symptoms, diagnoses, treatments, and patient outcomes.

## 🌟 Features

- **Natural Language Interface**: Ask complex clinical questions in plain English
- **Knowledge Graph**: Stores medical entities and their relationships in Neo4j
- **Document Processing**: Extract medical knowledge from clinical text documents
- **Interactive Visualization**: Explore graph statistics and query results
- **Citation Tracking**: Link answers back to original source documents

## 🏗️ Architecture

The application follows a modern layered architecture:

1. **UI Layer**: Streamlit-based interface for user interaction
2. **AI Layer**: Google Gemini models for language understanding and generation
3. **Processing Layer**: LangChain components for knowledge extraction and query processing
4. **Data Layer**: Neo4j graph database for knowledge storage

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Neo4j Database (accessible instance)
- Google API Key with access to Gemini models

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/breast-cancer-knowledge-explorer.git
cd breast-cancer-knowledge-explorer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file with your Neo4j credentials and Google API key
```

### Running the Application

```bash
streamlit run app.py
```

## 📊 How It Works

### Knowledge Graph Structure

The system organizes breast cancer information into an interconnected graph with node types including:

- **Patient**: Patient details and demographics
- **Symptom**: Clinical manifestations
- **Diagnosis**: Cancer types and classifications
- **Treatment**: Therapeutic approaches
- **Medication**: Drug information
- **Procedure**: Surgical and clinical interventions

Relationships between these entities (like `HAS_SYMPTOM`, `TREATED_WITH`) enable complex query traversals.

### Document Processing Pipeline

1. **Text Chunking**: Splits clinical documents into manageable segments
2. **Entity Extraction**: Identifies medical concepts and their properties
3. **Relationship Detection**: Establishes connections between entities
4. **Graph Integration**: Adds extracted knowledge to the Neo4j database

### Query Process

1. User submits a natural language question
2. The system translates it to a Cypher query using Google Gemini
3. Neo4j executes the query against the graph
4. Results are formatted into readable answers
5. Citations and sources are identified and presented

## 🔍 Example Queries

- "What symptoms are commonly associated with inflammatory breast cancer?"
- "Which treatments have shown the best outcomes for HER2-positive patients?"
- "How many patients with triple-negative breast cancer experienced lymphedema after treatment?"
- "What is the typical treatment pathway for stage II ductal carcinoma?"

## 🔄 Data Sources

The system can integrate breast cancer information from:

- Clinical research papers
- Medical guidelines and protocols
- De-identified patient records
- Clinical trial documentation

For comprehensive information about breast cancer, visit the [Indian Cancer Society](https://www.indiancancersociety.org/breast-cancer/index.html).

## 💻 Technical Details

### Key Components

- **Neo4jGraph**: Interface for database operations
- **LLMGraphTransformer**: Converts text to graph entities
- **GraphCypherQAChain**: Translates natural language to Cypher
- **GoogleGenerativeAI**: Powers language understanding and generation

### Functions

- `connect_neo4j()`: Establishes database connection
- `process_uploaded_file()`: Transforms documents into graph data
- `setup_qa_chain()`: Configures the question-answering system
- `display_graph_stats()`: Shows current database statistics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the document processing framework
- [Neo4j](https://neo4j.com/) for graph database technology
- [Google AI](https://ai.google/) for Gemini language models
- [Streamlit](https://streamlit.io/) for the web interface
- [Indian Cancer Society](https://www.indiancancersociety.org/) for breast cancer resources and information
