import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification
from remedies import get_remedy

st.set_page_config(
    page_title="AI Crop Disease Detector",
    page_icon="🌾",
    layout="centered",
)

MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

# Standard ImageNet-style preprocessing (matches how this model was trained).
# Done manually instead of via AutoImageProcessor to avoid a known
# preprocessor-resolution error on some transformers/Streamlit Cloud versions.
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource(show_spinner=False)
def load_model():
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return model


def predict(model, image: Image.Image, top_k: int = 4):
    pixel_values = PREPROCESS(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
    with torch.no_grad():
        logits = model(pixel_values).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    top_probs, top_idxs = torch.topk(probs, k=min(top_k, probs.shape[0]))
    id2label = model.config.id2label
    return [
        {"label": id2label[idx.item()], "score": prob.item()}
        for prob, idx in zip(top_probs, top_idxs)
    ]


def main():
    st.title("🌾 AI Crop Disease & Pest Detector")
    st.write(
        "Upload a photo of a plant leaf and get an instant AI diagnosis with a simple, "
        "actionable remedy — built for farmers who don't have easy access to an agronomist."
    )

    with st.spinner("Loading AI model... (only happens once)"):
        model = load_model()

    uploaded_file = st.file_uploader(
        "Upload a leaf photo (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded leaf", use_container_width=True)

        with st.spinner("Analyzing the leaf..."):
            predictions = predict(model, image)

        top = predictions[0]
        label = top["label"]
        confidence = top["score"] * 100

        disease_name, cause, remedy = get_remedy(label)

        st.markdown("---")
        if "healthy" in label.lower():
            st.success(f"✅ Diagnosis: **{disease_name}**  \nConfidence: {confidence:.1f}%")
        else:
            st.error(f"⚠️ Diagnosis: **{disease_name}**  \nConfidence: {confidence:.1f}%")

        st.subheader("Likely Cause")
        st.write(cause)

        st.subheader("Recommended Action")
        st.write(remedy)

        with st.expander("See other possible matches"):
            for pred in predictions[1:4]:
                alt_name, _, _ = get_remedy(pred["label"])
                st.write(f"- {alt_name} ({pred['score']*100:.1f}%)")

        st.caption(
            "⚠️ This is an AI-assisted first opinion, not a replacement for professional "
            "agricultural advice. For high-value crops or uncertain cases, please consult a "
            "local agricultural extension officer."
        )
    else:
        st.info("👆 Upload a leaf photo to get started.")


if __name__ == "__main__":
    main()
