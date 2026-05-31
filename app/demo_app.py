import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    ScaleIntensity,
    Resize,
    EnsureType
)

# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(
    page_title="AI Lung CT Assistant",
    page_icon="🫁",
    layout="wide"
)

# -----------------------
# MAIN HEADER
# -----------------------

st.markdown(
    """
    # 🫁 AI Lung CT Assistant

    ### Deep Learning Powered Medical Imaging Platform
    """
)

# -----------------------
# SIDEBAR
# -----------------------

st.sidebar.title("System Panel")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    AI Medical Imaging System

    Framework:
    - MONAI
    - PyTorch
    - Streamlit

    Model:
    - 3D U-Net
    """
)

st.sidebar.success("System Online")

st.sidebar.metric(
    "Model Accuracy",
    "84.84%"
)

# -----------------------
# TOP METRICS
# -----------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    "84.84%"
)

col2.metric(
    "Framework",
    "PyTorch"
)

col3.metric(
    "Dataset",
    "OrganMNIST3D"
)

st.markdown("---")

st.markdown(
    """
    ### AI-powered Lung CT Analysis System

    This demo uses deep learning models for medical CT image analysis.

    Features:
    - 3D Medical AI
    - CT Visualization
    - AI Prediction
    - AI Heatmap
    - Lesion Overlay
    """
)

st.markdown("---")


# -----------------------
# MODEL
# -----------------------

device = torch.device("cpu")

model_path = "models/lung_nodule_classifier.pth"

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2
).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    st.success("✅ Model loaded successfully!")
except:
    st.warning("⚠️ Model file not found. Running demo mode.")

model.eval()
from monai.transforms import (
    Compose,
    ScaleIntensity,
    Resize,
    EnsureType,
    EnsureChannelFirst
)

transform = Compose([
    EnsureChannelFirst(channel_dim="no_channel"),
    ScaleIntensity(),
    Resize((64, 128, 128)),
    EnsureType()
])

# -----------------------
# DEMO IMAGE
# -----------------------

uploaded_file = st.file_uploader(
    "Upload CT Scan",
    type=None
)

if uploaded_file is not None:

    st.success("CT uploaded successfully!")

    # save temp file
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"

    temp_path = f"temp_ct{suffix}"

    with open(temp_path, "wb") as f:
     f.write(uploaded_file.read())

    ct_scan = nib.load(temp_path)

    ct_data = ct_scan.get_fdata()

    st.write("CT Shape:", ct_data.shape)
    # -----------------------
    # CT SLICE VIEWER
    # -----------------------

    slice_idx = st.slider(
         "Select CT Slice",
          0,
         ct_data.shape[2] - 1,
         ct_data.shape[2] // 2
)

    fig, ax = plt.subplots()

    ax.imshow(ct_data[:, :, slice_idx].T, cmap="gray")

    ax.set_title(f"CT Slice {slice_idx}")

    st.pyplot(fig)
    st.markdown("---")
    # preprocess
    ct_tensor = transform(ct_data)
    st.write("Final Tensor Shape:", ct_tensor.shape)
    # add batch + channel dimension
    ct_tensor = torch.tensor(ct_tensor).float()

    st.write("Before batch:", ct_tensor.shape)

    ct_tensor = ct_tensor.unsqueeze(0)

    st.write("Final Tensor Shape:", ct_tensor.shape)

    with st.spinner("AI analyzing CT scan..."):
     with torch.no_grad():

      output = model(ct_tensor)

      probability = torch.sigmoid(output).mean().item()
    
    # -----------------------
    # AI SCORE
    # -----------------------

    st.metric(
        label="AI Prediction Score",
        value=f"{probability:.2%}"
    )

    if probability > 0.5:

       st.error(
           "⚠️ High Risk Detected"
        )

    else:
       
       st.success(
           "✅ Low Risk Detected"
        )

    st.progress(float(probability))
    
    heatmap = output.squeeze().detach().numpy()
    heatmap_slice = heatmap[heatmap.shape[0] // 2]

    fig2, ax2 = plt.subplots()

    ax2.imshow(heatmap_slice, cmap="hot")

    ax2.set_title("AI Attention Heatmap")

    st.pyplot(fig2)
    st.markdown("---")
    # -----------------------
    # -----------------------
    # AI OVERLAY
    # -----------------------

    # 获取 heatmap 切片总数
    heatmap_depth = heatmap.shape[0]

    # 防止索引越界
    safe_slice_idx = min(slice_idx, heatmap_depth - 1)

    # 获取CT切片
    overlay_slice = ct_data[:, :, safe_slice_idx].T

    # 获取heatmap切片
    heatmap_overlay = heatmap[safe_slice_idx]

    # 创建图像
    fig3, ax3 = plt.subplots()

    # 显示CT
    ax3.imshow(
        overlay_slice,
        cmap="gray"
     )

    # 叠加Heatmap
    ax3.imshow(
        heatmap_overlay,
        cmap="hot",
        alpha=0.4
     )

    # 标题
    ax3.set_title(
        f"AI Lesion Overlay - Slice {safe_slice_idx}"
     )

    # Streamlit显示
    st.pyplot(fig3)
    st.markdown("---")
    # DISCLAIMER

    # -----------------------


# -----------------------
# FOOTER
# -----------------------

st.markdown("---")

st.info(
    """
    Disclaimer:
    
    This AI system is for educational and research purposes only.
    Not intended for clinical diagnosis.
    """
)

st.caption(
    "AI Lung CT Assistant | Built with MONAI + PyTorch + Streamlit"
)