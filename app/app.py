import os
import streamlit as st
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
import torch.nn.functional as F

device = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../notebooks/lung_ct_unet.pth")

model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16,32,64,128), strides=(2,2,2), num_res_units=2).to(device)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

st.title("AI-Powered Lung CT Analysis")
uploaded_file = st.file_uploader("Upload Lung CT (.nii or .nii.gz)", type=["nii", "gz"])

if uploaded_file is not None:
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
    temp_path = "temp_ct" + suffix
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    ct_data = nib.load(temp_path).get_fdata().astype(np.float32)

    # 手动normalize
    ct_data = np.clip(ct_data, -1000, 400)
    ct_data = (ct_data - (-1000)) / (400 - (-1000))

    # 转tensor并resize到(64,128,128)
    ct_tensor = torch.from_numpy(ct_data).unsqueeze(0).unsqueeze(0)
    ct_tensor = F.interpolate(ct_tensor, size=(64,128,128), mode='trilinear', align_corners=False)

    with torch.no_grad():
        output = model(ct_tensor)
        pred = (ct_tensor > 0.8).float()

    ct_np = ct_tensor.cpu().numpy()
    pred_np = pred.cpu().numpy()

    slice_idx = ct_np.shape[-1] // 2
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(ct_np[0,0,:,:,slice_idx], cmap="bone")
    ax.imshow(pred_np[0,0,:,:,slice_idx],cmap="Reds",alpha=0.25)
    ax.set_title("AI Predicted Lesion")
    ax.axis("off")
    st.pyplot(fig)
    st.success("Analysis Completed!")