import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def classify_patient(note: str):

    prompt = f"""
You are a clinical triage assistant.

Classify the following patient note into ONE category:

CRITICAL – life-threatening condition
EMERGENCY – urgent medical attention needed
STABLE – condition stable or mild symptoms

Patient note:
{note}

Respond with ONLY one word:
CRITICAL, EMERGENCY, or STABLE.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    label = tokenizer.decode(outputs[0], skip_special_tokens=True).upper()

    if "CRITICAL" in label:
        return "CRITICAL"
    elif "EMERGENCY" in label:
        return "EMERGENCY"
    else:
        return "STABLE"