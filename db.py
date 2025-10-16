import os
from arango import ArangoClient
from langchain_community.graphs import ArangoGraph
from dotenv import load_dotenv

load_dotenv()
ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASS = os.getenv("ARANGO_PASS", "")

# ================= Database =================

try:
    client = ArangoClient(hosts=ARANGO_HOST)
    db = client.db("NeuThera", username=ARANGO_USER, password=ARANGO_PASS)
    print("Connected to ArangoDB:", db.name)

    arango_graph = ArangoGraph(db)
    print("ArangoGraph initialized successfully!")
except Exception as e:
    print(f"Warning: Could not connect to ArangoDB: {e}")
    print("Some features may not be available.")
    db = None
    arango_graph = None
