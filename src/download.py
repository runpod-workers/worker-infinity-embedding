from huggingface_hub import login, snapshot_download
import sys

token = sys.argv[1]

if not token:
    raise ValueError("JOORI_EMBEDDING_HUGGINGFACE_TOKEN environment variable is required")

try:
    login(token)
    print("🚀 Successfully logged in to Hugging Face Hub")
    snapshot_download(repo_id="Alibaba-NLP/gte-Qwen2-1.5B-instruct", local_dir="model")
    print("🚀 Model downloaded successfully")
except Exception as e:
    print(f"🚨 Download failed: {str(e)}")
    raise
