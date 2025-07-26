import openai
import yaml
import chromadb
from chromadb.config import Settings
from listings import read_listings
# from sentence_transformers import SentenceTransformer

# def get_embedding(text):
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # return model.encode(text).tolist()

def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    return embedding

if __name__ == "__main__":
    chroma_client = chromadb.Client(Settings(
        persist_directory="chroma_db"  # Path to your DB directory
    ))
    collection = chroma_client.create_collection(name="real_estate_listings")

    listings = read_listings()
    for i, listing in enumerate(listings):
        raw_yaml = yaml.dump(listing)
        emb = get_embedding(raw_yaml)
        collection.add(
            ids=[f"listing_{i}"],
            embeddings=[emb],
            documents=[raw_yaml],  # Store full listing for retrieval
            metadatas=[listing]    # Store as metadata for fast filtering
        )
        # Ignore Warning: Failed to send telemetry event collection_add: capture() takes 1 positional argument but 3 were given

    results = collection.get()     # Get everything (docs, embeddings, metadata, ids)
    print(results['ids'])          # List of all stored IDs
    print(results['metadatas'])    # Metadata dicts
    # print(results['documents'])  # Your stored text or YAML
