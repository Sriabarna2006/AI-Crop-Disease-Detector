# 🌾 AI Crop Disease & Pest Detector for Farmers

An AI-powered web app that lets a farmer upload a photo of a plant leaf and instantly get:
- The likely disease (or confirmation the plant is healthy)
- The likely cause
- A simple, actionable remedy

Built for the **Idea2Impact Online Hackathon 2026** — Theme: Sustainability & Social Impact (Rural Development).

## How it works

1. User uploads a leaf photo.
2. An image-classification AI model (pretrained on the PlantVillage dataset, 38 disease/healthy
   classes across crops like tomato, potato, corn, grape, apple, and more) predicts the most
   likely condition.
3. The app looks up a plain-language cause and remedy for the predicted disease and displays it.

**AI is the core of this app** — the diagnosis itself is produced by the model's prediction, not
hardcoded logic.

## Tech stack

- **Frontend/App**: [Streamlit](https://streamlit.io)
- **AI model**: Hugging Face `transformers` image-classification pipeline, using a MobileNetV2
  model fine-tuned on the PlantVillage dataset
- **Language**: Python

## Run locally

```bash
# 1. Clone this repo
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> Note: the first run will download the AI model from Hugging Face — this can take a minute or two.

## Deploy (Hugging Face Spaces — recommended, free)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Streamlit** as the Space SDK.
3. Upload `app.py`, `remedies.py`, and `requirements.txt` (or connect this GitHub repo).
4. The Space will build automatically and give you a public URL — that's your deployed link.

## Deploy (alternative — Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click "New app".
3. Point it at this repo and `app.py`.
4. Deploy — you'll get a public URL.

## Project structure

```
.
├── app.py           # Main Streamlit app
├── remedies.py       # Disease -> cause/remedy lookup table
├── requirements.txt  # Python dependencies
└── README.md
```

## Disclaimer

This tool provides an AI-assisted first opinion and is not a replacement for professional
agricultural advice. For high-value crops or uncertain cases, users are encouraged to consult a
local agricultural extension officer.
