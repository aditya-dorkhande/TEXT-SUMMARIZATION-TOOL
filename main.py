# Install required libraries
!pip install transformers

# Import necessary modules
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
model_name = "t5-base"  # Upgraded to "t5-base" for better quality
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text):
    """Function to summarize input text using T5 model."""
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    summary_ids = model.generate(
        input_ids, 
        max_length=70,   # Reduced summary length for better conciseness
        min_length=30,   # Minimum summary length
        length_penalty=2.5, 
        num_beams=5, 
        early_stopping=True,
        repetition_penalty=3.0  # Higher penalty to avoid repetition
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Input from user
input_text = input("\nEnter your lengthy text/article:\n").strip()

# Ensure input is long enough for summarization
if len(input_text.split()) > 50:
    summary = summarize_text(input_text)
    print("\nOriginal Text:\n", input_text)
    print("\nSummarized Text:\n", summary)
else:
    print("\n⚠️ Please enter a longer article (at least 50 words) for summarization.")
