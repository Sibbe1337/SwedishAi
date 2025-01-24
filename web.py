from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Skapa templates-mapp och lägg till denna HTML-fil där
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register")
async def register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/payment-success")
async def payment_success(request: Request):
    return templates.TemplateResponse("payment_success.html", {"request": request})

@app.get("/upgrade")
async def upgrade_page(request: Request):
    return templates.TemplateResponse("upgrade.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001) 