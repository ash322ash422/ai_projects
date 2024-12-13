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

# Insert documents into the Pinecone index
for doc in documents:
    embedding = get_embedding(doc['text'])# get embedding from openai model="text-embedding-ada-002"
    # print("..embedding=",embedding) # ..embedding= [0.008270886726677418, 0.023320067673921585, -0.007729992736130953, -0.02059505693614483, ..ommitted..] 
    print("len(embedding)=",len(embedding)) # 1536
    index.upsert([(doc['id'], embedding)],)

print("Documents have been inserted into the index.")
print("index.describe_index_stats()=\n",index.describe_index_stats())

print("###############################")
#search
query_str = 'medical 1'
query_embedding = get_embedding(query_str)
query_results = index.query(
    vector=query_embedding,
    top_k=1,
    include_values=True
)
print("query_results=",query_results)


"""
len(embedding)= 1536
len(embedding)= 1536
Documents have been inserted into the index.
index.describe_index_stats()=
 {'dimension': 1536,
 'index_fullness': 0.0,
 'namespaces': {'': {'vector_count': 5}},
 'total_vector_count': 5}
###############################
query_results= {'matches': [ ..ommitted..], 'namespace': 'ns1', 'usage': {'read_units': 1}}
"""