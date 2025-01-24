from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from queue import PriorityQueue
import heapq
import time
from enum import Enum
import numpy as np
from typing import Tuple, Dict, List

Base = declarative_base()

MODEL_NAME = "deepseek-ai/DeepSeek-V3"

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False

class ChatMessage(BaseModel):
    role: str  # "system", "user", eller "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False

class User(BaseModel):
    username: str
    email: str

class UserInDB(User):
    hashed_password: str
    is_premium: bool = False
    message_count: int = 0
    last_reset: datetime

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class SubscriptionPlan(BaseModel):
    plan: str
    price: float
    features: list[str]

class APIRequest(Base):
    __tablename__ = "api_requests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    endpoint = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prompt = Column(String)
    response = Column(String)
    status_code = Column(Integer)

@dataclass
class BatchItem:
    priority: int  # Lägre nummer = högre prioritet
    timestamp: float
    prompt: str
    user_id: int
    is_premium: bool

    def __lt__(self, other):
        if self.is_premium != other.is_premium:
            return self.is_premium  # Premium användare har företräde
        return self.timestamp < other.timestamp

class PromptTemplate(Enum):
    GENERAL = "general"
    CODE = "code"
    BUSINESS = "business"
    ACADEMIC = "academic"
    CREATIVE = "creative"

PROMPT_TEMPLATES = {
    PromptTemplate.GENERAL: {
        "system": """Du är en professionell AI-assistent byggd på DeepSeek-V3. 
                    Du hjälper användare med olika uppgifter på ett koncist och professionellt sätt.""",
        "context": "Kontext: Generell assistans och problemlösning",
        "format": "Format: Koncisa och tydliga svar"
    },
    PromptTemplate.CODE: {
        "system": """Du är en expertprogrammerare med djup kunskap om olika programmeringsspråk och tekniker.
                    Du hjälper till med kodning, debugging och arkitekturdesign.""",
        "context": "Kontext: Programmeringsfrågor och teknisk utveckling",
        "format": "Format: Kod med förklaringar och bästa praxis"
    },
    PromptTemplate.BUSINESS: {
        "system": """Du är en affärsstrategisk rådgivare med expertis inom företagsutveckling.
                    Du hjälper till med affärsstrategier, marknadsanalys och beslutsfattande.""",
        "context": "Kontext: Affärsutveckling och strategisk planering",
        "format": "Format: Strukturerade affärsrekommendationer"
    },
    PromptTemplate.ACADEMIC: {
        "system": """Du är en akademisk expert med bred kunskap inom olika forskningsområden.
                    Du hjälper till med forskningsfrågor, analys och akademiskt skrivande.""",
        "context": "Kontext: Akademisk forskning och utbildning",
        "format": "Format: Vetenskapligt grundade svar med referenser"
    },
    PromptTemplate.CREATIVE: {
        "system": """Du är en kreativ assistent med expertis inom skapande och design.
                    Du hjälper till med kreativt skrivande, design och idégenerering.""",
        "context": "Kontext: Kreativt skapande och design",
        "format": "Format: Inspirerande och kreativa förslag"
    }
}

class AdaptiveMoERouter:
    def __init__(self, num_experts: int, input_dim: int):
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.expert_weights = torch.ones(num_experts) / num_experts
        self.expert_performance = torch.zeros(num_experts)
        self.learning_rate = 0.01
        
    def route(self, inputs: torch.Tensor, task_type: str) -> Tuple[torch.Tensor, List[int]]:
        """Adaptiv routing baserad på input och uppgiftstyp"""
        # Beräkna task-specifika vikter
        task_weights = self._get_task_weights(task_type)
        
        # Kombinera med expertprestanda
        routing_weights = self.expert_weights * task_weights
        routing_weights = routing_weights / routing_weights.sum()
        
        # Välj experter baserat på input och vikter
        expert_scores = self._compute_expert_scores(inputs, routing_weights)
        selected_experts = self._select_top_experts(expert_scores)
        
        return routing_weights, selected_experts
    
    def update_performance(self, expert_ids: List[int], performance: float):
        """Uppdatera expertprestanda baserat på resultat"""
        for expert_id in expert_ids:
            self.expert_performance[expert_id] = (
                0.9 * self.expert_performance[expert_id] + 
                0.1 * performance
            )
            # Uppdatera vikter baserat på prestanda
            self.expert_weights = torch.softmax(self.expert_performance, dim=0)

class AIModel:
    def __init__(self):
        # Initiera prioritetskö för batch-hantering
        self.batch_queue = PriorityQueue()
        self.max_queue_size = 1000
        self.premium_batch_size = 16  # Större batch för premium
        self.regular_batch_size = 8   # Normal batch storlek

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            use_flash_attention_2=True,
            use_fp8_qdq=True
        )
        
        # DeepSeek-specifika optimeringar
        self.model.config.update({
            # Arkitektur-optimeringar
            "use_cache": True,
            "use_flash_attention": True,
            "use_sdpa": True,  # Scaled Dot Product Attention
            "use_merged_key_value": True,  # Optimerad KV cache
            
            # MoE (Mixture of Experts) inställningar
            "num_experts": 256,
            "num_active_experts": 37,
            "expert_capacity": 128,
            
            # Inferens-optimeringar
            "max_sequence_length": 128000,
            "rope_scaling": {"type": "dynamic", "factor": 2.0},
            "use_mtp": True,  # Multi-Token Prediction
            "mtp_window_size": 8,
            
            # Minnes-optimeringar
            "attention_dropout": 0.1,
            "gradient_checkpointing": True,
            "use_kernel_optimizations": True
        })
        
        # Aktivera optimeringar för batch-inferens
        if torch.cuda.is_available():
            self.model = torch.compile(self.model, mode="reduce-overhead")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        # Sätt upp optimerad tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Bättre för causalt språkmodellering

        # Initiera cache för KV-attention
        self._initialize_kv_cache()

        # Företagsspecifika inställningar
        self.company_configs = {}
        self.custom_prompts = {}

        # Initiera adaptiv MoE router
        self.moe_router = AdaptiveMoERouter(
            num_experts=self.model.config.num_experts,
            input_dim=self.model.config.hidden_size
        )
        
        # Prompt template hantering
        self.prompt_templates = PROMPT_TEMPLATES
        self.default_template = PromptTemplate.GENERAL

    def _initialize_kv_cache(self):
        """Initiera KV-cache för snabbare inferens"""
        self.kv_cache = {}
        if torch.cuda.is_available():
            # Reservera CUDA minne för cachen
            cache_size = 2 * self.model.config.num_hidden_layers * self.model.config.hidden_size
            self.kv_cache["reserved_memory"] = torch.cuda.memory_allocated() + cache_size

    async def add_to_batch_queue(self, prompt: str, user_id: int, is_premium: bool) -> None:
        """Lägg till en förfrågan i prioritetskön"""
        if self.batch_queue.qsize() >= self.max_queue_size:
            # Ta bort äldsta icke-premium förfrågan om kön är full
            items = []
            while not self.batch_queue.empty():
                items.append(self.batch_queue.get())
            items = [i for i in items if i.is_premium][:self.max_queue_size-1]
            for item in items:
                self.batch_queue.put(item)
        
        item = BatchItem(
            priority=0 if is_premium else 1,
            timestamp=time.time(),
            prompt=prompt,
            user_id=user_id,
            is_premium=is_premium
        )
        self.batch_queue.put(item)

    async def batch_generate(self, prompts: List[str], batch_size: int = 8) -> List[str]:
        """Generera svar för flera prompts samtidigt"""
        # Använd större batch för premium requests
        if all(p.is_premium for p in prompts):
            batch_size = self.premium_batch_size
        
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Använd gradient checkpointing för stora batches
            use_checkpoint = len(batch) > 8
            if use_checkpoint:
                self.model.gradient_checkpointing_enable()
            
            # Tokenisera batch
            inputs = self.tokenizer(batch, 
                                  padding=True, 
                                  truncation=True, 
                                  return_tensors="pt",
                                  max_length=self.model.config.max_sequence_length)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Optimera minne före generering
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Generera med batch
            outputs = self.model.generate(
                **inputs,
                max_length=self.model.config.max_sequence_length,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                num_experts=self.model.config.num_experts,
                num_active_experts=self.model.config.num_active_experts,
                use_mtp=True,
                mtp_window_size=8,
                use_kernel_optimizations=True,
                use_sdpa=True
            )
            
            # Dekodera och lägg till i responses
            batch_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            responses.extend(batch_responses)
            
            # Rensa cache efter varje batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return responses

    def optimize_model(self):
        """Optimera modellen för snabbare inferens"""
        if torch.cuda.is_available():
            # Aktivera CUDA grafer för snabbare exekvering
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Aktivera TensorFloat-32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    async def create_company_config(self, company_id: str, config: dict):
        """Skapa företagsspecifik konfiguration"""
        self.company_configs[company_id] = {
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 4096),
            "stop_sequences": config.get("stop_sequences", []),
            "custom_prompt_prefix": config.get("prompt_prefix", ""),
            "custom_prompt_suffix": config.get("prompt_suffix", ""),
            "allowed_models": config.get("allowed_models", ["deepseek-v3"]),
            "rate_limits": config.get("rate_limits", {
                "requests_per_minute": 60,
                "tokens_per_minute": 40000
            })
        }
    
    async def generate_with_company_config(
        self,
        company_id: str,
        prompt: str,
        **kwargs
    ):
        """Generera text med företagsspecifik konfiguration"""
        config = self.company_configs.get(company_id, {})
        
        # Applicera företagsspecifika inställningar
        full_prompt = (
            config.get("custom_prompt_prefix", "") +
            prompt +
            config.get("custom_prompt_suffix", "")
        )
        
        return await self.generate(
            prompt=full_prompt,
            temperature=kwargs.get("temperature", config.get("temperature", 0.7)),
            max_tokens=kwargs.get("max_tokens", config.get("max_tokens", 4096)),
            stop=kwargs.get("stop", config.get("stop_sequences", [])),
        ) 

    async def generate(self, prompt: str, **kwargs):
        # Identifiera lämplig promptmall baserat på innehåll
        template_type = kwargs.get("template", self._detect_template_type(prompt))
        template = self.prompt_templates[template_type]
        
        # Bygg formaterad prompt
        formatted_prompt = self._format_prompt(prompt, template)
        
        # Använd adaptiv MoE-routing
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        hidden_states = self.model.get_input_embeddings()(inputs["input_ids"])
        
        # Routing beslut
        routing_weights, selected_experts = self.moe_router.route(
            hidden_states, 
            str(template_type)
        )
        
        # Uppdatera generation config med valda experter
        generation_config = {
            **kwargs,
            "num_active_experts": len(selected_experts),
            "expert_ids": selected_experts,
            "routing_weights": routing_weights
        }

        try:
            outputs = self.model.generate(
                **inputs,
                **self._get_generation_config(generation_config),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # Utvärdera och uppdatera expertprestanda
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            performance = self._evaluate_response_quality(response)
            self.moe_router.update_performance(selected_experts, performance)

            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def _detect_template_type(self, prompt: str) -> PromptTemplate:
        """Detektera lämplig promptmall baserat på innehåll"""
        # Enkel innehållsbaserad detektion
        keywords = {
            PromptTemplate.CODE: ["kod", "programmering", "debug", "funktion", "class"],
            PromptTemplate.BUSINESS: ["företag", "strategi", "marknad", "affär"],
            PromptTemplate.ACADEMIC: ["forskning", "studie", "analys", "teori"],
            PromptTemplate.CREATIVE: ["design", "skapa", "kreativ", "idé"]
        }
        
        prompt = prompt.lower()
        for template, words in keywords.items():
            if any(word in prompt for word in words):
                return template
        return self.default_template

    def _format_prompt(self, prompt: str, template: dict) -> str:
        """Formatera prompt enligt vald mall"""
        return f"""<|im_start|>system
{template['system']}
{template['context']}
{template['format']}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""

    def _get_generation_config(self, generation_config: dict):
        """Uppdatera generation_config för adaptiv MoE-routing"""
        # Implementera logik för att uppdatera generation_config baserat på valda experter
        # Detta är en placeholder och bör implementeras korrekt baserat på dina specifikationer
        return generation_config

    def _evaluate_response_quality(self, response: str) -> float:
        """Evaluerer responsens kvalitet"""
        # Implementera logik för att evaluerera responsens kvalitet
        # Detta är en placeholder och bör implementeras korrekt baserat på dina specifikationer
        return 0.8  # Placeholder, bör ersättas med riktig evaluereringslogik

    def _get_task_weights(self, task_type: str) -> torch.Tensor:
        """Beräkna task-specifika vikter"""
        # Implementera logik för att beräkna task-specifika vikter baserat på task_type
        # Detta är en placeholder och bör implementeras korrekt baserat på dina specifikationer
        return torch.tensor([1.0] * self.model.config.num_experts)  # Placeholder, bör ersättas med riktig viktsberäkning

    def _compute_expert_scores(self, inputs: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        """Beräkna expertprestanda baserat på input och routing_weights"""
        # Implementera logik för att beräkna expertprestanda baserat på input och routing_weights
        # Detta är en placeholder och bör implementeras korrekt baserat på dina specifikationer
        return torch.matmul(inputs, routing_weights)  # Placeholder, bör ersättas med riktig prestandaberäkning

    def _select_top_experts(self, expert_scores: torch.Tensor) -> List[int]:
        """Välj topp experter baserat på expertprestanda"""
        # Implementera logik för att välja topp experter baserat på expertprestanda
        # Detta är en placeholder och bör implementeras korrekt baserat på dina specifikationer
        return torch.topk(expert_scores, k=self.model.config.num_active_experts).indices.tolist()  # Placeholder, bör ersättas med riktig toppexpertselektion

    def _get_generation_config(self, generation_config: dict):
        """Uppdatera generation_config för adaptiv MoE-routing"""
        # Implementera logik för att uppdatera generation_config baserat på valda experter
        # Detta är en placeholder och bör implementeras korrekt baserat på dina specifikationer
        return generation_config

    def _evaluate_response_quality(self, response: str) -> float:
        """Evaluerer responsens kvalitet"""
        # Implementera logik för att evaluerera responsens kvalitet
        # Detta är en placeholder och bör implementeras korrekt baserat på dina specifikationer
        return 0.8  # Placeholder, bör ersättas med riktig evaluereringslogik 