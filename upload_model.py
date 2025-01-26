from huggingface_hub import HfApi
import os
import torch

def create_config():
    """Create the model configuration file."""
    config = {
        "hidden_size": 768,  # DistilBERT's hidden size
        "num_classes": 2,    # Binary classification (real/fake)
        "model_name": "distilbert-base-uncased",
        "dropout": 0.1
    }
    
    # Save config file
    os.makedirs("models/final_model", exist_ok=True)
    config_path = "models/final_model/config.pt"
    torch.save(config, config_path, pickle_protocol=5)
    return config_path

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
    
    # Create and upload config file
    config_path = create_config()
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo="config.pt",
        repo_id=f"{username}/{repo_name}"
    )
    
    print(f"Config file uploaded successfully!")
    print(f"Your model URL: https://huggingface.co/{username}/{repo_name}/resolve/main/model.pt")
    print(f"Your config URL: https://huggingface.co/{username}/{repo_name}/resolve/main/config.pt")
    print("\nAdd the model.pt URL to your Render.com environment variables as MODEL_FILE_URL")

if __name__ == "__main__":
    upload_model() 