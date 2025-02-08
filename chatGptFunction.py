import torch
import torch.nn as nn

# Define a simple neural network model
class FlowerColorPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FlowerColorPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Predefined dataset of common flowers and colors
flower_colors = {
    "rose": "red",
    "tulip": "yellow",
    "sunflower": "yellow",
    "lily": "white",
    "daisy": "white",
    "orchid": "purple",
    "lavender": "purple",
    "hibiscus": "red",
    "marigold": "orange",
    "cherry blossom": "pink"
}

# Convert flower names to numeric tensors
flower_to_tensor = {name: torch.tensor([ord(c) for c in name], dtype=torch.float32) for name in flower_colors}
color_labels = list(set(flower_colors.values()))  # Unique color labels
color_to_index = {color: i for i, color in enumerate(color_labels)}
index_to_color = {i: color for color, i in color_to_index.items()}

# Define & load a pre-trained model (for demonstration, this is untrained)
input_size = max(len(name) for name in flower_colors)  # Max flower name length
hidden_size = 16
output_size = len(color_labels)

model = FlowerColorPredictor(input_size, hidden_size, output_size)
# Load a trained model if available
# model.load_state_dict(torch.load("flower_color_model.pth"))

# Function to predict flower color
def predict_flower_color(flower_name):
    if flower_name.lower() in flower_colors:
        return f"Most common color: {flower_colors[flower_name.lower()]}"  # Return from dataset

    # Convert input to tensor (padding to match input size)
    input_tensor = torch.zeros(input_size)
    name_tensor = torch.tensor([ord(c) for c in flower_name.lower()], dtype=torch.float32)
    input_tensor[:len(name_tensor)] = name_tensor  # Pad with zeros

    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_color = index_to_color[predicted_index]

    return f"Predicted color: {predicted_color}"

# Example Usage
print(predict_flower_color("rose"))  # Expected: red
print(predict_flower_color("orchid"))  # Expected: purple
print(predict_flower_color("random_flower"))  # Predicts color based on model
