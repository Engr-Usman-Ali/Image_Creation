import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline
from PIL import Image
import io

st.set_page_config(page_title="AI Image Studio", layout="wide")

# ----------------- Model Loader -----------------
@st.cache_resource
def load_models():
    models = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generation model
    try:
        models["generation"] = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    except Exception as e:
        st.error(f"Error loading generation model: {e}")
        models["generation"] = None

    # Upscale model
    try:
        models["upscale"] = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    except Exception as e:
        st.error(f"Error loading upscale model: {e}")
        models["upscale"] = None

    return models

models = load_models()

# ----------------- UI -----------------
st.title("🎨 AI Image Studio")

option = st.sidebar.radio("Choose Action", ["Generate Image", "Upscale Image", "History Gallery"])

# ----------------- Generate Image -----------------
if option == "Generate Image":
    st.subheader("Generate a New Image")

    prompt = st.text_area("Write your prompt:", placeholder="e.g., A futuristic city at sunset, cinematic lighting")

    if st.button("Generate"):
        if not prompt.strip():
            st.warning("Please write a prompt before generating.")
        elif models["generation"] is None:
            st.error("Image generation model not available.")
        else:
            try:
                # Truncate long prompts
                if len(prompt) > 500:
                    prompt = prompt[:500]
                    st.warning("Prompt too long. Truncated to 500 characters.")

                with st.spinner("Creating image..."):
                    image = models["generation"](prompt=prompt, num_inference_steps=20).images[0]

                # Show image
                st.image(image, caption="Generated Image", use_container_width=True)

                # Save in session
                st.session_state["last_image"] = image
                st.session_state["last_prompt"] = prompt

                if "gallery" not in st.session_state:
                    st.session_state["gallery"] = []
                st.session_state["gallery"].append(("Generated", image))

                # Download button
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button("Download Image", buf.getvalue(), "generated.png", "image/png")

                # Extra actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Regenerate"):
                        st.session_state["regenerate"] = True
                with col2:
                    if st.button("Modify Prompt"):
                        st.session_state["modify"] = True

            except Exception as e:
                st.error(f"Error generating image: {e}")

    # Regenerate
    if "regenerate" in st.session_state and st.session_state.get("last_prompt"):
        try:
            with st.spinner("Regenerating..."):
                image = models["generation"](prompt=st.session_state["last_prompt"], num_inference_steps=20).images[0]
            st.image(image, caption="Regenerated Image", use_container_width=True)

            st.session_state["last_image"] = image
            st.session_state["gallery"].append(("Regenerated", image))
            st.session_state.pop("regenerate")
        except Exception as e:
            st.error(f"Error regenerating: {e}")

    # Modify prompt
    if "modify" in st.session_state and st.session_state.get("last_prompt"):
        new_prompt = st.text_area("Modify your previous prompt:", st.session_state["last_prompt"])
        if st.button("Generate from Modified Prompt"):
            try:
                if len(new_prompt) > 500:
                    new_prompt = new_prompt[:500]
                    st.warning("Prompt too long. Truncated to 500 characters.")

                with st.spinner("Generating from modified prompt..."):
                    image = models["generation"](prompt=new_prompt, num_inference_steps=20).images[0]

                st.image(image, caption="Modified Image", use_container_width=True)

                st.session_state["last_image"] = image
                st.session_state["last_prompt"] = new_prompt
                st.session_state["gallery"].append(("Modified", image))
                st.session_state.pop("modify")

            except Exception as e:
                st.error(f"Error generating modified image: {e}")

# ----------------- Upscale Image -----------------
elif option == "Upscale Image":
    st.subheader("Upscale Your Image")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if st.button("Upscale"):
        if uploaded_file is None and "last_image" not in st.session_state:
            st.warning("Please upload an image or generate one first!")
        elif models["upscale"] is None:
            st.error("Upscaler model not loaded.")
        else:
            try:
                with st.spinner("Upscaling..."):
                    img = Image.open(uploaded_file) if uploaded_file else st.session_state["last_image"]

                    # Always provide dummy prompt
                    upscaled_image = models["upscale"](prompt="upscale", image=img).images[0]

                st.image(upscaled_image, caption="Upscaled Image", use_container_width=True)

                buf = io.BytesIO()
                upscaled_image.save(buf, format="PNG")
                st.download_button("Download Upscaled Image", buf.getvalue(), "upscaled.png", "image/png")

                if "gallery" not in st.session_state:
                    st.session_state["gallery"] = []
                st.session_state["gallery"].append(("Upscaled", upscaled_image))

            except Exception as e:
                st.error(f"Error upscaling image: {e}")

# ----------------- History Gallery -----------------
elif option == "History Gallery":
    st.subheader("📸 Your Image Gallery")

    if "gallery" in st.session_state and st.session_state["gallery"]:
        for idx, (label, img) in enumerate(st.session_state["gallery"]):
            st.image(img, caption=f"{label} #{idx+1}", use_container_width=True)
    else:
        st.info("No images in gallery yet. Generate or upscale to see history.")
