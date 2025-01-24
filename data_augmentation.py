from transformers import pipeline
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import torch
import random
from typing import List

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