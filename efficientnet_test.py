import torch
from datasets import load_dataset
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
import time
import psutil

def run_efficientnet():
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB

    # Load an example image dataset
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]  # Get the first image

    # Load EfficientNet-B7 preprocessor and model
    preprocessor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b7")
    model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b7")

    # Preprocess the image
    inputs = preprocessor(image, return_tensors="pt")

    # Run inference without computing gradients
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted class
    predicted_label = logits.argmax(-1).item()
    print("Predicted Label:", model.config.id2label[predicted_label])

    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
    
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Memory Usage: {end_memory - start_memory:.2f} MB")

if __name__ == "__main__":
    run_efficientnet()
