from pinecone import Pinecone

PINECONE_API_KEY = "73df847a-9909-4a70-b03c-a1fc19655006"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-docs") # NOTE: make sure this exist on the server. Dimension = 1536

#########################################
import openai, os
 
OPENAI_API_KEY="key_here"
openai.api_key = OPENAI_API_KEY
print("OPENAI_API_KEY=",OPENAI_API_KEY)

def get_embedding(text): # creates embedding of dim=1536
    response = openai.Embedding.create( 
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

documents = [
    {"id": "doc1", "text": "Medical document content for Doc 1. Side effects of Aspirin are headache, and ulcers"},
    {"id": "doc2", "text": "Medical document content for Doc 2. Medications should be taken under supervision of doctors"}
]

# Insert documents into the Pinecone index with metadata
for doc in documents:
    embedding = get_embedding(doc['text'])  # Get embedding from OpenAI model
    print("len(embedding)=", len(embedding))  # Should print 1536
    index.upsert([{
        'id': doc['id'],
        'values': embedding,
        'metadata': {'text': doc['text']}  # Include document text as metadata
    }])

print("Documents have been inserted into the index.")
print("index.describe_index_stats()=\n",index.describe_index_stats())

print("###############################")
#search
query_str = 'aspirin effects'
query_embedding = get_embedding(query_str)
query_results = index.query(
    vector=query_embedding,
    top_k=1,
    include_metadata=True  # Include metadata in the results to retrieve the text
)

print("query_results=",query_results)
# Print the query results and the matching text
print("Query Results:")
for match in query_results['matches']:
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']}")
    print(f"Text: {match['metadata']['text']}")  # Retrieve the text from metadata


"""
len(embedding)= 1536
len(embedding)= 1536
Documents have been inserted into the index.
index.describe_index_stats()=
 {'dimension': 1536,
 'index_fullness': 0.0,
 'namespaces': {'': {'vector_count': 5}, 'ns1': {'vector_count': 2}},
 'total_vector_count': 7}
###############################
query_results= {'matches': [{'id': 'doc1',
              'metadata': {'text': 'Medical document content for Doc 1. Side '
                                   'effects of Aspirin are headache, and '
                                   'ulcers'},
              'score': 0.874879241,
              'values': []}],
 'namespace': '',
 'usage': {'read_units': 6}}
Query Results:
ID: doc1
Score: 0.874879241
Text: Medical document content for Doc 1. Side effects of Aspirin are headache, and ulcers
"""