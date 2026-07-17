import streamlit as st
from PIL import Image
from transformers import pipeline
from remedies import get_remedy

st.set_page_config(
    page_title="AI Crop Disease Detector",
    page_icon="🌾",
    layout="centered",
)

MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"


@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("image-classification", model=MODEL_NAME)


def main():
    st.title("🌾 AI Crop Disease & Pest Detector")
    st.write(
        "Upload a photo of a plant leaf and get an instant AI diagnosis with a simple, "
        "actionable remedy — built for farmers who don't have easy access to an agronomist."
    )

    with st.spinner("Loading AI model... (only happens once)"):
        classifier = load_model()

    uploaded_file = st.file_uploader(
        "Upload a leaf photo (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded leaf", use_container_width=True)

        with st.spinner("Analyzing the leaf..."):
            predictions = classifier(image)

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
