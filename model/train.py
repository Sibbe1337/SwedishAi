from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import logging
import os
import warnings
from accelerate import Accelerator

# Ignorera deprecation varningar
warnings.filterwarnings("ignore", category=FutureWarning)

# Sätt upp loggning
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initiera accelerator
accelerator = Accelerator()

def prepare_dataset():
    """Ladda och förbered svensk textdata"""
    try:
        # Ladda den nedladdade datan
        dataset = load_dataset(
            'text',
            data_files={'train': 'model/data/swedish_text.txt'}
        )
        
        # Dela upp i träning och validering
        dataset = dataset['train'].train_test_split(test_size=0.1)
        
        return dataset
    except Exception as e:
        logger.error(f"Fel vid datainläsning: {str(e)}")
        raise

def train_model():
    """Träna språkmodellen"""
    try:
        # Använd birgermoell's svenska GPT-modell
        model_name = "birgermoell/swedish-gpt"  # Svensk GPT2 tränad på Wikipedia
        
        logger.info(f"Laddar modell {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Lägg till pad_token om den saknas
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Lade till pad_token (använder eos_token)")
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Flytta modell till rätt device via accelerator
        model = accelerator.prepare(model)
        
        # Ladda dataset
        dataset = prepare_dataset()
        
        # Tokenisera data
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors=None  # Viktigt för batch-processing
            )
        
        logger.info("Tokeniserar dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        # Träningskonfiguration
        training_args = TrainingArguments(
            output_dir="./model/checkpoints",
            per_device_train_batch_size=2,  # Mindre batch size för att spara minne
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            logging_dir="./model/logs",
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000,
            evaluation_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            gradient_accumulation_steps=8,  # Öka för att kompensera mindre batch size
            fp16=True  # Använd mixed precision för bättre prestanda
        )
        
        # Initiera trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test']
        )
        
        # Träna modellen
        logger.info("Startar träning...")
        trainer.train()
        
        # Spara modellen
        output_dir = "./model/swedish-ai-model-final"
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Modell sparad i {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Fel vid träning: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Först ladda ner data om det inte redan är gjort
        if not os.path.exists("model/data/swedish_text.txt"):
            logger.info("Ingen data hittad, kör prepare_data.py först")
            exit(1)
            
        model_path = train_model()
        logger.info("Träning slutförd framgångsrikt!")
    except Exception as e:
        logger.error(f"Träning misslyckades: {str(e)}") 