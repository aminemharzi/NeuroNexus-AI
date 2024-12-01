import numpy as np
import random
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import re
import streamlit as st

# Load your dataset
df = pd.read_csv("full_Chat_data.csv")


# Function to clean text (removes special characters, lowers text, and removes extra whitespaces)
def clean_text(text):
    if isinstance(text, str):  # Check if the text is a string
        text = text.lower()  # Lowercase the text
        text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespaces
    else:
        text = ""  # Replace NaN or non-string with empty string
    return text


# Apply cleaning to both questions and answers
df["Questions"] = df["Questions"].apply(clean_text)
df["Answers"] = df["Answers"].apply(clean_text)

# Create a list of dictionaries where each document consists of a question-answer pair
documents = []
for index, row in df.iterrows():
    document = {"question": row["Questions"], "answer": row["Answers"]}
    documents.append(document)


# Number of clients in Federated Learning
num_clients = 3


# Function to simulate local training by encoding questions into embeddings (local model)
def local_training(documents, model):
    # Encode the questions into embeddings using Sentence-BERT
    question_embeddings = model.encode([doc["question"] for doc in documents])
    return np.array(question_embeddings).astype("float32")


# Federated Learning round simulation
def federated_learning_round(documents, num_clients, model):
    # Split the dataset across clients (randomly)
    client_data = [
        random.sample(documents, len(documents) // num_clients)
        for _ in range(num_clients)
    ]

    # Initialize a list to hold the embeddings for each client
    client_embeddings = []

    # Simulate local training on each client
    for client in range(num_clients):
        print(f"Client {client+1} is training...")
        client_embeddings.append(local_training(client_data[client], model))

    # Aggregate the client embeddings (simulating model aggregation)
    print("Aggregating client models...")
    global_embeddings = np.mean(client_embeddings, axis=0)

    # Use the aggregated embeddings as the new global model for the next round
    return global_embeddings


# Initialize the Sentence-BERT model (a pre-trained model for sentence embeddings)
model = SentenceTransformer("all-MiniLM-L6-v2")  # A smaller model for efficiency

# Simulate federated learning rounds
rounds = 1
global_embeddings = None
for round_num in range(rounds):
    print(f"\nFederated Learning Round {round_num + 1}")
    global_embeddings = federated_learning_round(documents, num_clients, model)

# Save global_embeddings to a file
np.save("global_embeddings.npy", global_embeddings)
print("Global embeddings saved to 'global_embeddings.npy'")

# Later, you can load the global embeddings
loaded_embeddings = np.load("global_embeddings.npy")
print("Global embeddings loaded from 'global_embeddings.npy'")

# Create a FAISS index
index = faiss.IndexFlatL2(global_embeddings.shape[1])  # L2 distance (Euclidean)
index.add(global_embeddings)


def retrieve_answer(query):
    # Encode the query to obtain its embedding
    query_embedding = model.encode([query])[0]
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

    # Search for the most similar questions in FAISS
    _, indices = index.search(
        query_embedding, k=1
    )  # We want the top 1 most similar question

    # Retrieve the most similar question and its answer
    result = documents[indices[0][0]]

    return result["answer"]
