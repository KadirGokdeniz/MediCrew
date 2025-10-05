from transformers import AutoTokenizer, AutoModel
import torch

print("BioBERT Model Download and Test")
print("="*50)

# Device check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

if device == "cpu":
    print("WARNING: No GPU found, will use CPU (will be slow)")
else:
    print(f"GPU found: {torch.cuda.get_device_name(0)}")

# Model name
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

print(f"\nDownloading model: {model_name}")
print("First download ~500MB, will take a few minutes...")

# Download model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)
model.eval()

print("Model successfully downloaded!")
print(f"  Dimension: 768")

# Simple test
print("\nRunning simple test...")
test_text = "Metformin is used for diabetes treatment"

# Tokenize
inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate embedding
with torch.no_grad():
    outputs = model(**inputs)

# Get result
embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

print(f"Test successful!")
print(f"  Embedding size: {len(embedding)}")
print(f"  First 5 values: {embedding[:5]}")

print("\n" + "="*50)
print("Model is ready!")
print("\nModel saved in:")
print("~/.cache/huggingface/hub/")
print("\nWon't download again on next run.")