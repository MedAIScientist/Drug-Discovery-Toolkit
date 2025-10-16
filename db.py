import os
from arango import ArangoClient
from langchain_community.graphs import ArangoGraph
from dotenv import load_dotenv

load_dotenv()
ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASS = os.getenv("ARANGO_PASS", "openSesame")

# ================= Database =================

try:
    client = ArangoClient(hosts=ARANGO_HOST)

    # First connect to _system database to check/create NeuThera database
    sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASS)

    # Check if NeuThera database exists, create if not
    if not sys_db.has_database("NeuThera"):
        sys_db.create_database("NeuThera")
        print("Created NeuThera database")

    # Now connect to NeuThera database
    db = client.db("NeuThera", username=ARANGO_USER, password=ARANGO_PASS)
    print("Connected to ArangoDB:", db.name)

    # Create required collections if they don't exist
    required_collections = ["drugs", "proteins", "drug_protein_links"]
    for collection_name in required_collections:
        if not db.has_collection(collection_name):
            if collection_name == "drug_protein_links":
                db.create_collection(collection_name, edge=True)
            else:
                db.create_collection(collection_name)
            print(f"Created collection: {collection_name}")

    # Create graph if it doesn't exist
    if not db.has_graph("drug_protein_graph"):
        graph = db.create_graph(
            "drug_protein_graph",
            edge_definitions=[
                {
                    "edge_collection": "drug_protein_links",
                    "from_vertex_collections": ["drugs"],
                    "to_vertex_collections": ["proteins"],
                }
            ],
        )
        print("Created drug_protein_graph")

    arango_graph = ArangoGraph(db)
    print("ArangoGraph initialized successfully!")
except Exception as e:
    print(f"Warning: Could not connect to ArangoDB: {e}")
    print("Some features may not be available.")
    db = None
    arango_graph = None
