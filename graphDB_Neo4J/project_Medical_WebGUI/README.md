
Real-time GraphRAG (Graph Retrieval-Augmented Generation) application that leverages the power of [LangChain](https://python.langchain.com/), [Neo4j](https://neo4j.com/), and OpenAI's GPT models to extract knowledge from documents and enable natural language querying over a graph database.

## Overview

Graphy v1 allows users to upload a PDF document, which is then processed to extract entities and relationships using OpenAI's GPT models (e.g., `gpt-4o` and `text-ada-002`). The extracted information is stored in a Neo4j graph database. Users can then interact with the graph in real-time by asking natural language questions, which are converted into Cypher queries to retrieve answers from the graph.

## Features

- **Real-time GraphRAG**: Extracts knowledge from documents and allows real-time querying.
- **Modular and Configurable**: Users can set their own credentials for OpenAI and Neo4j.
- **Natural Language Interface**: Ask questions in plain English and get answers from the graph database.

### Prerequisites

- Python 3.11 or higher
- An OpenAI API key with access to GPT models (e.g., `gpt-4o` or `ada 002`)
- A Neo4j database instance (remote)
- Streamlit

#To run (may have to run twice):
   streamlit run main.py

Sample question : List all the medications.
