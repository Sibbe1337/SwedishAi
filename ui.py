import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging

logger = logging.getLogger(__name__)

# Modellkonfiguration
MODEL_MAX_LENGTH = 2048  # GPT-SW3 stödjer längre sekvenser
TEMPERATURE = 0.7
TOP_P = 0.9

# Ladda modell
model_path = "./models/finetuned-swedish-gpt"
model_name = model_path if os.path.exists(model_path) else "AI-Sweden-Models/gpt-sw3-126m"
model_type = "finetuned" if os.path.exists(model_path) else "base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
logger.info(f"Använder {model_type} GPT-SW3 modell")

# Lägg till pad_token om den saknas
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MODEL_MAX_LENGTH)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Ett fel uppstod: {str(e)}"

# Skapa Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Skriv din text här...",
        label="Din text"
    ),
    outputs=gr.Textbox(
        label="Genererad text",
        lines=5
    ),
    title=f"SwedishAI Chat (GPT-SW3 {model_type})",
    description="""En svensk AI-chattbot tränad på:
- Wikipedia-artiklar om Sverige och svensk kultur
- Data från SCB
- Andra svenska texter

Baserad på AI Sweden's GPT-SW3 126M modell.""",
    examples=[
        ["Hej! Hur mår du?"],
        ["Berätta en kort historia om Sverige."],
        ["Vad är typiskt svenskt?"],
        ["Kan du beskriva svensk kultur?"],
        ["Berätta om svenska traditioner"],
    ]
)

if __name__ == "__main__":
    iface.launch(server_name="localhost", server_port=8002, share=False)