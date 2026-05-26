import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from medmnist import INFO
from medmnist.dataset import NoduleMNIST3D

# -----------------------------
# Page Title
# -----------------------------
st.title("AI Lung CT Analysis Demo")

st.write(
    "3D Medical AI Classification using PyTorch"
)

# -----------------------------
# Define Model
# -----------------------------
class Simple3DCNN(nn.Module):

    def __init__(self):

        super(Simple3DCNN, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv3d(
                1, 8,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(
                8, 16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(2)
        )

        self.fc_layers = nn.Sequential(

            nn.Linear(
                16 * 7 * 7 * 7,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2
            )
        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = x.view(x.size(0), -1)

        x = self.fc_layers(x)

        return x

# -----------------------------
# Load Model
# -----------------------------
model = Simple3DCNN()

model.load_state_dict(
    torch.load(
        "../models/lung_nodule_classifier.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# -----------------------------
# Load Dataset
# -----------------------------
data_flag = "nodulemnist3d"

info = INFO[data_flag]

DataClass = getattr(
    __import__("medmnist"),
    info["python_class"]
)

test_dataset = DataClass(
    split="test",
    download=True
)

# -----------------------------
# Select Sample
# -----------------------------
sample_idx = st.slider(
    "Select CT Sample",
    0,
    len(test_dataset)-1,
    0
)

image, label = test_dataset[sample_idx]

image_tensor = (
    torch.tensor(image)
    .unsqueeze(0)
    .float()
)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():

    output = model(image_tensor)

    _, predicted = torch.max(output, 1)

# -----------------------------
# Visualization
# -----------------------------
slice_idx = 14

fig, ax = plt.subplots(figsize=(5,5))

ax.imshow(
    image[0, :, :, slice_idx],
    cmap="gray"
)

ax.set_title(
    f"True: {label[0]} | Predicted: {predicted.item()}"
)

ax.axis("off")

st.pyplot(fig)

# -----------------------------
# Result
# -----------------------------
if predicted.item() == label[0]:

    st.success("Prediction Correct")

else:

    st.error("Prediction Incorrect")
