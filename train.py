from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
import torch
import logging
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from transformers import pipeline
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Använd GPU/MPS om tillgängligt
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Använder device: {device}")
is_gpu = device.type == "cuda"

# Nuvarande konstanter
MAX_LENGTH = 512
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.01
BASE_MODEL = "deepseek-ai/deepseek-coder-33b-instruct"

# Uppdaterade hyperparametrar med förklaringar
class TrainingConfig:
    def __init__(self):
        # Grundläggande parametrar
        self.max_length = 512
        self.num_epochs = 15        # Ökat för att tillåta bättre konvergens
        
        # Batchstorlek och inlärning
        self.batch_size = 4         # Ökat för stabilare träning
        self.gradient_accumulation_steps = 4  # Effektiv batchstorlek = 16
        
        # Inlärningshastighet och schemaläggning
        self.learning_rate = 2e-5   # Något högre för snabbare initial inlärning
        self.min_learning_rate = 1e-6  # Lägsta LR för scheduler
        self.warmup_ratio = 0.1     # 10% av stegen för uppvärmning
        
        # Regularisering
        self.weight_decay = 0.02
        self.dropout = 0.1
        
        # Early stopping
        self.patience = 3
        self.min_delta = 0.01

class TrainingMetrics:
    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.epochs: List[int] = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.no_improvement_count = 0
        
    def update(self, epoch: int, train_loss: float, val_loss: float) -> bool:
        """Uppdatera metriker och kontrollera early stopping"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if val_loss < self.best_val_loss - EARLY_STOPPING_THRESHOLD:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered efter {epoch} epoker")
                return True
        return False
    
    def save(self, output_dir: str):
        """Spara träningsmetriker och plotta resultat"""
        metrics_path = os.path.join(output_dir, 'training_metrics.npy')
        np.save(metrics_path, {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        })
        
        # Plotta träningskurvor
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, 'b-', label='Träningsförlust')
        plt.plot(self.epochs, self.val_losses, 'r-', label='Valideringsförlust')
        plt.axvline(x=self.best_epoch, color='g', linestyle='--', label='Bästa epoch')
        plt.title('Tränings- och Valideringsförlust')
        plt.xlabel('Epoch')
        plt.ylabel('Förlust')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'training_loss.png'))
        plt.close()

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.01, path='checkpoint.pt'):
        """
        Args:
            patience (int): Antal epoker att vänta innan träningen stoppas
            min_delta (float): Minsta förändring som räknas som förbättring
            path (str): Sökväg där bästa modellen sparas
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            return False
            
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
    
    def save_checkpoint(self, model: torch.nn.Module):
        """Spara modellen när vi hittar en bättre valideringsförlust"""
        torch.save(model.state_dict(), self.path)

class DataAugmenter:
    def __init__(self, model_name: str = "AI-Sweden-Models/gpt-sw3-126m"):
        self.generator = pipeline("text-generation", model=model_name)
        self.back_translator = naw.BackTranslationAug(
            from_model_name='Helsinki-NLP/opus-mt-sv-en',
            to_model_name='Helsinki-NLP/opus-mt-en-sv'
        )
        
    def generate_synthetic(self, prompt: str, n_samples: int = 5) -> List[str]:
        """Generera nya texter baserat på prompt"""
        samples = []
        
        for _ in range(n_samples):
            output = self.generator(
                prompt,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9
            )
            samples.append(output[0]["generated_text"])
            
        return samples
    
    def augment_text(self, text: str) -> List[str]:
        """Använd olika augmenteringstekniker"""
        augmented = []
        
        # Back-translation
        aug_texts = self.back_translator.augment(text)
        augmented.extend(aug_texts)
        
        # Synonym replacement
        aug = naw.SynonymAug(aug_src='wordnet', lang='swe')
        aug_texts = aug.augment(text)
        augmented.extend(aug_texts)
        
        return augmented

class AugmentedDataset:
    def __init__(self, base_dataset, augmenter: DataAugmenter):
        self.base_dataset = base_dataset
        self.augmenter = augmenter
        
    def __len__(self):
        return len(self.base_dataset) * 2  # Original + augmenterad
        
    def __getitem__(self, idx):
        base_idx = idx // 2
        is_augmented = idx % 2 == 1
        
        if is_augmented:
            text = self.base_dataset[base_idx]["text"]
            augmented = self.augmenter.augment_text(text)[0]
            return {"text": augmented}
        
        return self.base_dataset[base_idx]

def prepare_augmented_dataset(
    file_path: str = "data/training_data.txt",
    use_augmentation: bool = True
) -> Dataset:
    """Förbered dataset med augmentering"""
    
    # Ladda grunddataset
    dataset = load_dataset('text', data_files={'train': file_path})
    
    if use_augmentation:
        # Initiera augmenterare
        augmenter = DataAugmenter()
        
        # Skapa augmenterat dataset
        augmented_dataset = AugmentedDataset(dataset['train'], augmenter)
        
        # Tokenisera
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Lade till pad_token")
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=MAX_LENGTH,
                return_tensors=None
            )
        
        logger.info("Tokeniserar dataset...")
        tokenized_dataset = augmented_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        logger.info("Tokenisering klar")
        
        return tokenized_dataset
    
    return dataset

def get_training_args(config: TrainingConfig, output_dir: str) -> TrainingArguments:
    """Skapa optimerade träningsargument"""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        
        # Batch och gradient ackumulering
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,  # Större eval batch
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Inlärningsschema
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine_with_restarts",  # Cosine med restarts
        warmup_ratio=config.warmup_ratio,
        
        # Regularisering
        weight_decay=config.weight_decay,
        
        # Utvärdering och sparande
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        
        # Optimeringar
        fp16=is_gpu,  # Använd mixed precision på GPU
        gradient_checkpointing=True,  # Spara minne
        
        # Diverse
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

class CustomTrainer(Trainer):
    def __init__(self, early_stopping, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stopping = early_stopping
        
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = super().compute_loss(model, inputs, return_outputs)
        return loss if return_outputs else loss[0]
        
    def log(self, logs: Dict):
        super().log(logs)
        if "eval_loss" in logs:
            val_loss = logs.get("eval_loss", 0)
            if self.early_stopping(val_loss, self.model):
                self.state.should_training_stop = True
                logger.info(f"Early stopping triggered. Bästa val_loss: {self.early_stopping.best_loss:.4f}")

def train_model(output_dir: str = "./models/finetuned-swedish-gpt"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'best_model.pt')
        
        # Initiera early stopping
        early_stopping = EarlyStopping(
            patience=3,
            min_delta=0.01,
            path=checkpoint_path
        )
        
        # Ladda modell och tokenizer
        logger.info("Laddar modell och tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Förbered dataset
        dataset = prepare_augmented_dataset()
        
        # Träningskonfiguration
        training_args = get_training_args(TrainingConfig(), output_dir)
        
        trainer = CustomTrainer(
            early_stopping=early_stopping,
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
        )
        
        logger.info("Startar träning...")
        trainer.train()
        
        # Ladda bästa modellen
        model.load_state_dict(torch.load(checkpoint_path))
        
        # Spara slutgiltig modell och tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Träning slutförd. Bästa val_loss: {early_stopping.best_loss:.4f}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Fel vid träning: {str(e)}")
        raise

class MixupTrainer(CustomTrainer):
    def __init__(self, *args, alpha=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.training:
            # Implementera mixup
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = inputs["input_ids"].size(0)
            index = torch.randperm(batch_size)
            
            mixed_input_ids = lam * inputs["input_ids"] + (1 - lam) * inputs["input_ids"][index]
            mixed_labels = lam * inputs["labels"] + (1 - lam) * inputs["labels"][index]
            
            inputs = {
                "input_ids": mixed_input_ids,
                "labels": mixed_labels,
                "attention_mask": inputs["attention_mask"]
            }
            
        return super().compute_loss(model, inputs, return_outputs)

if __name__ == "__main__":
    train_model() 