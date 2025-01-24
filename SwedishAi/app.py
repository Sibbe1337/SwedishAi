from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
import torch
import logging
import os
from database import Conversation, Base, sessionmaker, SessionLocal, User, Subscription
from datetime import datetime, timedelta
import logging.handlers
from models import (
    Token,
    UserInDB,
    SubscriptionPlan,
    CompletionRequest,
    ChatCompletionRequest,
    ChatMessage
)
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List
import asyncio
from collections import deque
import uuid
import time

# Databas dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Queue för batch-processing
message_queue = deque()
BATCH_SIZE = 8
QUEUE_TIMEOUT = 0.1  # sekunder

async def generate_response(message: str, history: List[dict]) -> str:
    try:
        # DeepSeek-specifik prompt formatering
        system_prompt = """Du är en professionell AI-assistent som hjälper användare med olika uppgifter. 
                         Du är byggd på DeepSeek-V3, en avancerad språkmodell med 671B parametrar och 
                         specialiserad på både naturligt språk och kod. Svara koncist och professionellt."""
        
        # Formatera historiken enligt DeepSeek format
        formatted_history = ""
        for msg in history:
            formatted_history += f"""<|im_start|>user
 {msg['user']}
 <|im_end|>
 <|im_start|>assistant
 {msg['assistant']}
 <|im_end|>
 """
        
        # Bygg fullständig prompt
        prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
{formatted_history}
<|im_start|>user
{message}
<|im_end|>
<|im_start|>assistant
Jag är en hjälpsam AI-assistent baserad på DeepSeek-V3, en state-of-the-art språkmodell med 671B parametrar.
Jag kommer att svara på dina frågor på ett professionellt och hjälpsamt sätt.
"""
        
        # Lägg till meddelandet i prioritetskön
        await model.add_to_batch_queue(
            prompt=prompt,
            user_id=current_user.id,
            is_premium=current_user.is_premium
        )
        
        # Vänta kort tid för att samla fler meddelanden
        await asyncio.sleep(QUEUE_TIMEOUT)
        
        # Processa batch med prioritet
        if not model.batch_queue.empty():
            # Ta ut batch från kön
            batch_items = []
            while not model.batch_queue.empty() and len(batch_items) < model.premium_batch_size:
                item = model.batch_queue.get()
                batch_items.append(item)
            
            # Formatera prompts för batch
            prompts = [
                f"""<|im_start|>user
{item.prompt}
<|im_end|>
<|im_start|>assistant
Jag är en hjälpsam AI-assistent baserad på DeepSeek-V3, en state-of-the-art språkmodell med 671B parametrar.
Jag kommer att svara på dina frågor på ett professionellt och hjälpsamt sätt.
""" for item in batch_items
            ]
            
            # Generera svar för hela batchen
            responses = await model.batch_generate(prompts, batch_size=batch_size)
            
            # Returnera svaret för det aktuella meddelandet
            response_index = next(i for i, item in enumerate(batch_items) if item.prompt == prompt)
            return responses[response_index].strip()
        
        # Om timeout och ingen batch, processa enskilt meddelande
        prompt = f"""<|im_start|>user
{message}
<|im_end|>
<|im_start|>assistant
Jag är en hjälpsam AI-assistent baserad på DeepSeek-V3, en state-of-the-art språkmodell med 671B parametrar.
Jag kommer att svara på dina frågor på ett professionellt och hjälpsamt sätt.
"""
        
        # Konfigurera generation med Multi-Token Prediction
        response = model.generate(
            prompt,
            max_length=128000,  # 128K kontext
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            num_experts=256,  # MoE arkitektur
            num_active_experts=37,  # 37B aktiva parametrar
            use_mtp=True,  # Aktivera Multi-Token Prediction
            mtp_window_size=8,  # Förutsäg 8 tokens samtidigt
            use_cache=True,  # Aktivera KV-cache
            attention_dropout=0.1,  # Lägg till dropout för bättre generalisering
            stop_tokens=["<|im_end|>"]
        )
        
        # Optimera minne efter generering
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response.strip()
    except Exception as e:
        logger.error(f"Fel vid textgenerering: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Ett fel uppstod vid textgenerering"
        )

# Sätt upp loggning
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sätt upp roterad filloggning
handler = logging.handlers.RotatingFileHandler(
    'app.log',
    maxBytes=1024*1024,
    backupCount=5
)
logger.addHandler(handler)

# Använd CPU om MPS/CUDA inte är tillgängligt
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Använder device: {device}")

# Initiera DeepSeek modell
MODEL_NAME = "deepseek-ai/DeepSeek-V3"
logger.info(f"Initierar DeepSeek-V3 modell...")

# DeepSeek konfiguration
MAX_LENGTH = 128000  # 128K kontext
TEMPERATURE = 0.7
TOP_P = 0.95

# Initiera modell och tokenizer
model = AIModel()  # Använder vår AIModel klass från models.py
logger.info("DeepSeek-V3 modell initierad med följande konfiguration:")
logger.info(f"- Max kontext: {MAX_LENGTH} tokens")
logger.info(f"- Antal experter: {model.model.config.num_experts}")
logger.info(f"- Aktiva experter: {model.model.config.num_active_experts}")

logger.info("DeepSeek-V3 modell redo för inferens")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Säkerhetskonfiguration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Definiera credentials_exception
credentials_exception = HTTPException(
    status_code=401,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(email: str):
    db = SessionLocal()
    return db.query(User).filter(User.email == email).first()

async def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# Request models
class PromptRequest(BaseModel):
    prompt: str

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = get_user(payload.get("sub"))
        if user is None:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception

async def check_message_limit(user: User = Depends(get_current_user)):
    if not user.is_premium:
        # Reset count if it's a new month
        if datetime.utcnow() - user.last_reset > timedelta(days=30):
            user.message_count = 0
            user.last_reset = datetime.utcnow()
        
        if user.message_count >= 100:
            raise HTTPException(
                status_code=429,
                detail="Message limit reached. Please upgrade to premium."
            )

@app.post("/generate")
async def generate_text(
    request: PromptRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Hämta användaren från den aktuella sessionen
        user = db.query(User).filter(User.id == current_user.id).first()
        
        # Kontrollera användarens begränsningar
        if not user.is_premium and user.message_count >= 5:
            raise HTTPException(
                status_code=429,
                detail="Gratis användare är begränsade till 5 meddelanden per dag"
            )
        
        logger.info(f"Genererar text för prompt: {request.prompt[:20]}...")
        
        # Generera text
        response = await generate_response(request.prompt, [])
        logger.info(f"Genererade {len(response)} tecken")
        
        # Uppdatera användarens meddelanderäknare
        user.message_count += 1
        db.commit()
        
        return {"generated_text": response}
        
    except Exception as e:
        logger.error(f"Fel vid textgenerering: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Ett fel uppstod vid textgenerering"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": str(device)
    }

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register(user: UserInDB):
    db = SessionLocal()
    if get_user(user.email):
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    db_user = User(
        email=user.email,
        hashed_password=get_password_hash(user.password),
        is_premium=False,
        message_count=0,
        last_reset=datetime.utcnow()
    )
    db.add(db_user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/upgrade")
async def upgrade_to_premium(
    user: User = Depends(get_current_user)
):
    db = SessionLocal()
    db_user = get_user(user.email)
    db_user.is_premium = True
    
    subscription = Subscription(
        user_id=db_user.id,
        plan="pro",
        active=True,
        started_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(days=30)
    )
    
    db.add(subscription)
    db.commit()
    return {"message": "Upgraded to premium successfully"}

@app.get("/user/info")
async def get_user_info(user: User = Depends(get_current_user)):
    return {
        "email": user.email,
        "is_premium": user.is_premium,
        "message_count": user.message_count,
        "last_reset": user.last_reset
    }

# Sätt upp templates
templates = Jinja2Templates(directory="templates")

# Montera statiska filer
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    try:
        token = request.cookies.get("access_token")
        if token:
            user = await get_current_user(token)
        else:
            user = None
    except:
        user = None

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user
    })

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    try:
        token = request.cookies.get("access_token")
        if token:
            user = await get_current_user(token)
        else:
            user = None
    except:
        user = None

    return templates.TemplateResponse("about.html", {
        "request": request,
        "user": user
    })

@app.get("/docs", response_class=HTMLResponse)
async def docs_page(request: Request):
    return templates.TemplateResponse("docs.html", {
        "request": request,
        "user": None
    })

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    return templates.TemplateResponse("privacy.html", {
        "request": request,
        "user": None
    })

@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    return templates.TemplateResponse("terms.html", {
        "request": request,
        "user": None
    })

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "user": current_user
    })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "user": None
    })

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/upgrade", response_class=HTMLResponse)
async def upgrade_page(request: Request):
    return templates.TemplateResponse("upgrade.html", {"request": request})

@app.get("/training-stats", response_class=HTMLResponse)
async def training_stats_page(request: Request):
    return templates.TemplateResponse("training_stats.html", {"request": request})

@app.get("/api/training-stats")
async def get_training_stats():
    from analyze_training import load_metrics, analyze_training
    
    try:
        model_dir = "./models/finetuned-swedish-gpt"
        
        # Kontrollera om katalogen finns
        if not os.path.exists(model_dir):
            logger.error(f"Modellkatalog saknas: {model_dir}")
            raise HTTPException(
                status_code=404,
                detail="Träningsstatistik är inte tillgänglig - modellen har inte tränats än"
            )
        
        # Kontrollera om metrics-filen finns
        metrics_path = os.path.join(model_dir, 'training_metrics.npy')
        if not os.path.exists(metrics_path):
            logger.error(f"Metrics-fil saknas: {metrics_path}")
            raise HTTPException(
                status_code=404,
                detail="Träningsstatistik saknas - kör train.py först"
            )
        
        # Ladda och analysera metriker
        try:
            metrics = load_metrics(model_dir)
            analysis, recommendations = analyze_training(metrics)
            
            logger.info("Träningsstatistik laddad framgångsrikt")
            return {
                "metrics": metrics,
                "analysis": analysis,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Fel vid laddning av träningsstatistik: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Kunde inte analysera träningsstatistik: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Oväntat fel i get_training_stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Ett oväntat fel uppstod vid hämtning av träningsstatistik"
        )

@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        response = await model.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop
        )
        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{
                "text": response,
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(request.prompt),
                "completion_tokens": len(response),
                "total_tokens": len(request.prompt) + len(response)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        messages = request.messages
        response = await model.chat(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {
                "prompt_tokens": sum(len(m["content"]) for m in messages),
                "completion_tokens": len(response),
                "total_tokens": sum(len(m["content"]) for m in messages) + len(response)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))