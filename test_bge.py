from sentence_transformers import SentenceTransformer
import time

start = time.time()
print("Loading BAAI/bge-m3...")
model = SentenceTransformer("BAAI/bge-m3")
load_time = time.time() - start

print(f"Model loaded in {load_time:.2f}s")
vec = model.encode("This is a test sentence.")
print(f"Vector dimension: {len(vec)}")
