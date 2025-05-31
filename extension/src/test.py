from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI (replace with your actual project and region)
aiplatform.init(project="camels-453517", location="us-central1")

# Test the embedding model
try:
    embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    test_text = ["Hello, this is a test for embedding."]
    embeddings = embedding_model.get_embeddings(test_text)
    print("Embedding model is working!")
    print("First 5 values of embedding:", embeddings[0].values[:5])
except Exception as e:
    print("Embedding model test failed:", e)

# Test the generative model
try:
    context_model = GenerativeModel("gemini-2.0-flash")
    response = context_model.generate_content("What is the capital of France?")
    print("Generative model is working!")
    print("Model output:", response.text)
except Exception as e:
    print("Generative model test failed:", e)