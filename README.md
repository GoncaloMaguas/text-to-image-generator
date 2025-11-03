# Text-to-Image Generator using Deep Learning

A Python project that generates images from text prompts using state-of-the-art deep learning models. This project leverages Hugging Face Diffusers, PyTorch, and other machine learning libraries to create high-quality, AI-generated images through a simple web interface built with **Streamlit**.

## Features

- Generate images from natural language text prompts
- Easy-to-use web interface with Streamlit
- Supports GPU acceleration for faster inference
- Compatible with multiple deep learning backends (PyTorch, TensorFlow optional)
- Save generated images locally

## Installation

1. Clone the repository:

```bash
git clone https://github.com/GoncaloMaguas/text-to-image-generator.git
cd text-to-image-generator
```
2. Create a virtual environment and activate it:
On Windows 
```
python -m venv venv
venv\Scripts\activate
```

On Linux/macOS
```
python -m venv venv
source venv/bin/activate
```
3. Install dependencies:
 ```
   pip install -r requirements.txt
```

4. Run the Streamlit app:

```
streamlit run Text_To_Image.py
```

## Dependencies

Key libraries used in this project:

```bash
torch                # PyTorch deep learning backend
tensorflow           # Optional backend
diffusers            # Hugging Face diffusion models
transformers         # Hugging Face text encoders
streamlit            # Web interface
numpy                # Numerical operations
matplotlib           # Plotting and visualization
Pillow               # Image processing





