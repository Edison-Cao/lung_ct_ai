import streamlit as st
import torch
import torch.nn.functional as F  
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tempfile  
import os        

from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    ScaleIntensity,
    Resize,
    EnsureType,
    EnsureChannelFirst
)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="AI Lung CT Assistant",
    page_icon="🫁",
    layout="wide"
)

st.markdown(
    """
    # 🫁 AI Lung CT Assistant
    ### Deep Learning Powered Medical Imaging Platform
    """
)

# =========================
# Sidebar
# =========================
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

# =========================
# Model Setup
# =========================
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

model_loaded = False
try:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_loaded = True
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.warning("⚠️ Model file not found. Running in demo mode (random weights — results are meaningless).")
except RuntimeError as e:
    st.error(f"❌ Model architecture mismatch: {e}")
    st.warning("Running in demo mode (random weights — results are meaningless).")
except Exception as e:
    st.error(f"❌ Unexpected error loading model: {e}")
    st.warning("Running in demo mode (random weights — results are meaningless).")

model.eval()

if model_loaded:
    st.sidebar.success("Model Status: ✅ Loaded")
else:
    st.sidebar.warning("Model Status: ⚠️ Demo Mode")

st.sidebar.metric("Model Accuracy", "84.84%*")
st.sidebar.caption("*Placeholder — not from real evaluation")

# =========================
# Preprocessing Pipeline
# =========================
transform = Compose([
    EnsureChannelFirst(channel_dim="no_channel"),
    ScaleIntensity(),
    Resize((64, 128, 128)),
    EnsureType()
])

# =========================
# Page Header Metrics
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("Framework", "PyTorch")
col2.metric("Model", "3D U-Net")
col3.metric("Dataset", "OrganMNIST3D*")
st.caption("*OrganMNIST3D is an abdominal organ dataset used to validate the pipeline. Not real lung CT data.")

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

# =========================
# Patient Clinical Information
# =========================
st.header("Patient Clinical Information")

col1, col2 = st.columns(2)

with col1:
    patient_age = st.number_input(
        "Patient Age",
        min_value=1,
        max_value=120,
        value=45
    )
    patient_gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    smoking_history = st.selectbox(
        "Smoking History",
        ["No", "Former Smoker", "Current Smoker"]
    )
    symptoms = st.multiselect(
        "Symptoms",
        ["Cough", "Chest Pain", "Shortness of Breath", "Fever", "Fatigue"]
    )

st.markdown("---")

# =========================
# State Variables
# =========================
probability = None
output = None
heatmap = None
heatmap_resized_np = None  
slice_idx = 0
ct_data = None

# =========================
uploaded_file = st.file_uploader(
    "Upload CT Scan (.nii or .nii.gz)",
    type=["nii", "gz"]
)

if uploaded_file is not None:
    left_col, right_col = st.columns([1, 2])

    # KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Patient Age", patient_age)
    with kpi2:
        st.metric("Smoking Risk", smoking_history)
    with kpi3:
        st.metric("Model Status", "Loaded ✅" if model_loaded else "Demo ⚠️")
    with kpi4:
        st.metric("CT Status", "Uploaded")

    st.markdown("---")
    st.success("CT uploaded successfully!")

    # -----------------------
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        try:
            ct_scan = nib.load(temp_path)
            ct_data = ct_scan.get_fdata()
        except Exception as e:
            st.error(f"❌ Failed to read CT file: {e}. Please ensure you uploaded a valid NIfTI file.")
            st.stop()
        finally:
            os.unlink(temp_path)

    except Exception as e:
        st.error(f"❌ File upload error: {e}")
        st.stop()

    # -----------------------
    # CT SLICE VIEWER
    # -----------------------
    with right_col:
        slice_idx = st.slider(
            "Select CT Slice",
            0,
            ct_data.shape[2] - 1,
            ct_data.shape[2] // 2
        )

        fig, ax = plt.subplots()
        ax.imshow(ct_data[:, :, slice_idx].T, cmap="gray")
        ax.set_title(f"CT Slice {slice_idx}")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------
    # Preprocessing
    # -----------------------
    ct_tensor = transform(ct_data)

    ct_tensor = ct_tensor.float()
    ct_tensor = ct_tensor.unsqueeze(0)  # (1, 1, 64, 128, 128)

    # -----------------------
    # Model Inference
    # -----------------------
    with st.spinner("AI analyzing CT scan..."):
        with torch.no_grad():
            output = model(ct_tensor)

            sigmoid_output = torch.sigmoid(output)

            threshold = 0.5
            probability = (sigmoid_output > threshold).float().mean().item()

    # -----------------------
    # AI SCORE
    # -----------------------
    with left_col:
        st.metric(
            label="AI Activation Ratio",  
            value=f"{probability:.2%}",
            help="Proportion of CT volume above activation threshold (0.5). "
                 "Higher value = more voxels flagged as potentially abnormal."
        )

        if probability > 0.1:
            st.error("⚠️ Elevated AI Activation Detected")
        else:
            st.success("✅ Low Activation — No Significant Abnormality Flagged")

        st.progress(float(min(probability * 5, 1.0)))
        
        # -----------------------
        # AI CLINICAL SUMMARY
        # -----------------------
        st.markdown("---")
        st.subheader("AI Clinical Summary")
        st.caption("⚠️ Rule-based summary — not AI-generated clinical advice")

        if probability is None:
            st.info("📂 Please upload a CT scan to generate the summary.")
        else:
            summary = []

            if probability > 0.1:
                summary.append("Elevated AI activation detected in CT volume.")
            else:
                summary.append("No significant AI activation detected.")

            if smoking_history == "Current Smoker":
                summary.append("Current smoking history may increase pulmonary risk.")
            elif smoking_history == "Former Smoker":
                summary.append("Former smoking history noted.")

            if len(symptoms) > 0:
                summary.append(f"Reported symptoms: {', '.join(symptoms)}.")

            summary.append("Further clinical review by a qualified radiologist is recommended.")
            summary.append("This AI output is for research/educational purposes only and should not be used for clinical diagnosis.")

            for item in summary:
                st.write("•", item)

    # -----------------------
    # HEATMAP + OVERLAY
    # -----------------------
    if output is not None and ct_data is not None:

        st.markdown("---")

        heatmap = output.squeeze().detach().numpy()  # shape: (64, 128, 128)

        heatmap_tensor_for_resize = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()

        target_size = (ct_data.shape[0], ct_data.shape[1], ct_data.shape[2])

        heatmap_resized_tensor = F.interpolate(
            heatmap_tensor_for_resize,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        heatmap_resized_np = heatmap_resized_tensor.squeeze().numpy()

        # -----------------------
        # HEATMAP DISPLAY
        # -----------------------
        with right_col:
            heatmap_display_slice = heatmap_resized_np[:, :, heatmap_resized_np.shape[2] // 2].T

            fig2, ax2 = plt.subplots()
            ax2.imshow(heatmap_display_slice, cmap="hot")
            ax2.set_title("AI Activation Heatmap (Spatially Aligned)")
            ax2.axis("off")
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown("---")

        # -----------------------
        # AI OVERLAY
        # -----------------------
        with right_col:
            safe_slice_idx = min(slice_idx, heatmap_resized_np.shape[2] - 1)

            overlay_slice = ct_data[:, :, safe_slice_idx].T

            heatmap_overlay = heatmap_resized_np[:, :, safe_slice_idx].T

            fig3, ax3 = plt.subplots()
            ax3.imshow(overlay_slice, cmap="gray")
            ax3.imshow(heatmap_overlay, cmap="hot", alpha=0.4)
            ax3.set_title(f"AI Lesion Overlay — Slice {safe_slice_idx} (Spatially Aligned)")
            ax3.axis("off")
            st.pyplot(fig3)
            plt.close(fig3)

        st.markdown("---")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.info(
    """
    Disclaimer:

    This AI system is for educational and research purposes only.
    Not intended for clinical diagnosis.
    All AI outputs should be interpreted by qualified medical professionals.
    """
)
st.caption("AI Lung CT Assistant | Built with MONAI + PyTorch + Streamlit")