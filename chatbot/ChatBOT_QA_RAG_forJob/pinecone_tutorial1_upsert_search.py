from pinecone import Pinecone

PINECONE_API_KEY = "73df847a-9909-4a70-b03c-a1fc19655006"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("quickstart") # NOTE: make sure this exist on the server
################################
#upsert
index.upsert(
    vectors=[
        {
            "id": "vec1", 
            "values": [1.0, 1.5], 
            "metadata": {"genre": "drama"}
        }, {
            "id": "vec2", 
            "values": [2.0, 1.0], 
            "metadata": {"genre": "action"}
        }, {
            "id": "vec3", 
            "values": [0.1, 0.3], 
            "metadata": {"genre": "drama"}
        }, {
            "id": "vec4", 
            "values": [1.0, -2.5], 
            "metadata": {"genre": "action"}
        }
    ],
    namespace= "ns1"
)

print("index.describe_index_stats()=\n",index.describe_index_stats())
print("##################################")
#search

query_results1 = index.query(
    namespace="ns1",
    vector=[1.0, 1.5],
    top_k=3,
    include_values=True
)

query_results2 = index.query(
    namespace="ns1",
    vector=[1.0,-2.5],
    top_k=3,
    include_values=True
)

print("query_results1=",query_results1)
print("query_results2=",query_results2)
print("##################################")
#cleanup
# pc.delete_index(index_name)

"""
index.describe_index_stats()=
 {'dimension': 2,
 'index_fullness': 0.0,
 'namespaces': {'ns1': {'vector_count': 4}},
 'total_vector_count': 4}
##################################
query_results1= {'matches': [{'id': 'vec1', 'score': 1.0, 'values': [1.0, 1.5]},
             {'id': 'vec3', 'score': 0.96476382, 'values': [0.1, 0.3]},
             {'id': 'vec2', 'score': 0.868243158, 'values': [2.0, 1.0]}],
 'namespace': 'ns1',
 'usage': {'read_units': 6}}
query_results2= {'matches': [{'id': 'vec4', 'score': 1.0, 'values': [1.0, -2.5]},
             {'id': 'vec2', 'score': -0.0830454826, 'values': [2.0, 1.0]},
             {'id': 'vec1', 'score': -0.566528857, 'values': [1.0, 1.5]}],
 'namespace': 'ns1',
 'usage': {'read_units': 6}}
##################################

"""