import torch
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms

# Define the transformation for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")

def predict_image_text(image_path, model_path):
    # Load the model
    model = ImageToTextModel(swin_encoder, t5_decoder).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)

    return predicted_text

# Example usage
image_path = 'path/to/your/image.jpg'
model_path = 'path/to/your/saved_model/model.pth'
predicted_text = predict_image_text(image_path, model_path)
print("Predicted Text:", predicted_text)
