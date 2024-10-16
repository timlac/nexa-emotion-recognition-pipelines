import os
import numpy as np
import faiss
from deepface import DeepFace

def extract_face_embedding(image_path, model_name='VGG-Face'):
    # Extract embedding using DeepFace
    embedding = DeepFace.represent(img_path=image_path, model_name=model_name, enforce_detection=False)
    return np.array(embedding[0]["embedding"])


def create_faiss_index(embedding_dim=128):
    # Create a FAISS index with L2 distance (euclidean)
    index = faiss.IndexFlatL2(embedding_dim)
    return index


def add_embeddings_to_index(index, embeddings):
    # Convert embeddings to float32 (required by Faiss)
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)


def process_and_index_faces(frames_dir, model_name='VGG-Face', threshold=0.5):
    # Create the Faiss index
    embedding_list = []
    frame_paths = []

    # Loop through frames and extract embeddings
    for frame_file in os.listdir(frames_dir):
        frame_path = os.path.join(frames_dir, frame_file)
        print(frame_path)

        if os.path.isfile(frame_path):
            embedding = extract_face_embedding(frame_path, model_name=model_name)
            embedding_list.append(embedding)
            frame_paths.append(frame_path)

    index = create_faiss_index(embedding_dim=embedding_list[0].shape[0])
    # Add all embeddings to the index
    add_embeddings_to_index(index, embedding_list)

    return index, np.array(embedding_list), frame_paths


def find_unique_faces(index, embeddings, frame_paths, threshold=0.5):
    # Find all unique faces by querying the index
    unique_faces = []
    unique_frame_paths = []

    # Track which embeddings have already been matched
    processed_indices = set()

    for i, embedding in enumerate(embeddings):
        if i not in processed_indices:
            # Search the index for similar embeddings
            D, I = index.search(np.array([embedding]), k=10)  # k=10 nearest neighbors
            unique_faces.append(embedding)
            unique_frame_paths.append(frame_paths[i])

            # Mark matched indices as processed based on distance threshold
            for j in range(len(I[0])):
                if D[0][j] < threshold:
                    processed_indices.add(I[0][j])

    return unique_faces, unique_frame_paths


# Example usage
frames_dir = "../out/frame_data/temp_frames_small"
index, embeddings, frame_paths = process_and_index_faces(frames_dir)

v =  embeddings[0].reshape(1, -1)

v = np.float32(v)

D, I = index.search(v, )

# Find unique faces and get paths of frames with unique faces
# unique_faces, unique_frame_paths = find_unique_faces(index, embeddings, frame_paths)

# print(f"Found {len(unique_faces)} unique faces.")
