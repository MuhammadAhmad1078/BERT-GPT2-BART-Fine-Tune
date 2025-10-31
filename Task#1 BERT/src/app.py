import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================
# 🔹 Load Fine-tuned Model
# =====================================================
MODEL_PATH = "./model/bert_sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# =====================================================
# 🔹 Streamlit UI Setup
# =====================================================
st.set_page_config(page_title="Sentiment Classifier", page_icon="💬", layout="centered")

st.title("💬 Real-Time Sentiment Classifier (BERT)")
st.markdown("Fine-tuned BERT model for analyzing **customer feedback** sentiment.")

# Textbox for user input
user_input = st.text_area("📝 Enter text to analyze:", height=120, placeholder="Type your feedback here...")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before prediction.")
    else:
        # Tokenize and predict
        tokens = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokens)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()

        # Display result
        st.markdown(f"### 🧠 Predicted Sentiment: **{label_map[pred_label]}**")
        st.progress(int(confidence * 100))
        st.caption(f"Confidence: {confidence:.2f}")

# Footer
st.markdown("---")
st.caption("Model: BERT-base-uncased | Fine-tuned on customer feedback dataset (97 samples)")
