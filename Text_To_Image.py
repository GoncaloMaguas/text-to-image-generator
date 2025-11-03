import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# -------------------------------------------------------------
# INITIAL CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(page_title="Text-to-Image Generator", page_icon="🖼️", layout="centered")
st.title("🧠 Text-to-Image Generator")
st.write("Create original images from textual descriptions using **Stable Diffusion**.")

# -------------------------------------------------------------
# USER INPUT
# -------------------------------------------------------------
prompt = st.text_input("💬 Write a description:", "a futuristic city at sunset")
num_images = st.slider("How many images to generate?", min_value=1, max_value=3, value=1)

# -------------------------------------------------------------
# GENERATE IMAGES
# -------------------------------------------------------------
if st.button("🎨 Generate Image"):
    st.write("⏳ Loading model (may take some time the first time)...")

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set correct data type
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype
    ).to(device)

    st.success("Model loaded successfully!")
    st.write("🖼️ Generating image...")

    # Generate images
    if device == "cuda":
        # Autocast only on GPU
        with torch.autocast("cuda"):
            images = pipe([prompt] * num_images).images
    else:
        images = pipe([prompt] * num_images).images

    # Display and save
    for i, img in enumerate(images):
        st.image(img, caption=f"Image {i+1}", use_container_width=True)
        img.save(f"output_{i+1}.png")
        st.download_button(
            label=f"📥 Download Image {i+1}",
            data=open(f"output_{i+1}.png", "rb").read(),
            file_name=f"output_{i+1}.png",
            mime="image/png"
        )

    st.success("✅ Image generated successfully!")

# -------------------------------------------------------------
# INSTRUCTIONS
# -------------------------------------------------------------
st.markdown("""
---
### ℹ️ How to use:
1. Write a description (e.g., *“a cyberpunk robot playing guitar in space”*).  
2. Click **Generate Image**.  
3. Download your creation!
---
""")
