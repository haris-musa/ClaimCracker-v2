from huggingface_hub import HfApi
import os

def upload_model():
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Your Hugging Face username - replace this
    username = "harismusa"
    
    # Create a new repository
    repo_name = "claimcracker-model"
    repo_url = api.create_repo(
        repo_id=f"{username}/{repo_name}",
        private=True,
        exist_ok=True
    )
    
    # Upload the model file
    api.upload_file(
        path_or_fileobj="models/final_model/model.pt",
        path_in_repo="model.pt",
        repo_id=f"{username}/{repo_name}"
    )
    
    print(f"Model uploaded successfully!")
    print(f"Your model URL: https://huggingface.co/{username}/{repo_name}/resolve/main/model.pt")
    print("\nAdd this URL to your Render.com environment variables as MODEL_FILE_URL")

if __name__ == "__main__":
    upload_model() 