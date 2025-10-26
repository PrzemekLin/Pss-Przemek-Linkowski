import os
import json
import threading
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ------------------------------
# Konfiguracja
# ------------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "products.json")
LOCK = threading.Lock()

# Wczytaj zmienne środowiskowe
load_dotenv()
API_KEY = os.getenv("API_KEY", "secret")


# ------------------------------
# Funkcje do obsługi bazy
# ------------------------------
def _ensure_db():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"products": [], "next_id": 1}, f, indent=2)


def load_db():
    _ensure_db()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


# ------------------------------
# Modele Pydantic
# ------------------------------
class ProductIn(BaseModel):
    name: str
    price: float
    category: str


class ProductOut(ProductIn):
    id: int


# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Lab2 – Products CRUD API", version="0.1.0")

# ------------------------------
# Middleware CORS
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ------------------------------
# Middleware API Key Guard
# ------------------------------
@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    # Endpointy chronione
    protected_paths = ["/admin/", "/products"]

    if any(request.url.path.startswith(p) for p in protected_paths):
        provided_key = request.headers.get("X-API-Key")
        if provided_key != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized (missing/invalid X-API-Key)"}
            )
    # Jeśli klucz poprawny lub endpoint nie chroniony → kontynuuj
    response = await call_next(request)
    return response


# ------------------------------
# Endpointy publiczne
# ------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# Endpoint admina
@app.get("/admin/secret")
def admin_secret():
    return {"ok": True, "msg": "Welcome, admin."}


# ------------------------------
# CRUD /products
# ------------------------------
@app.get("/products", response_model=List[ProductOut])
def list_products():
    db = load_db()
    return db["products"]


@app.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: int):
    db = load_db()
    for p in db["products"]:
        if p["id"] == product_id:
            return p
    raise HTTPException(status_code=404, detail="Product not found")


@app.post("/products", response_model=ProductOut, status_code=201)
def create_product(product: ProductIn):
    with LOCK:
        db = load_db()
        new_id = db.get("next_id", 1)
        rec = {"id": new_id, **product.dict()}
        db["products"].append(rec)
        db["next_id"] = new_id + 1
        save_db(db)
        return rec


@app.put("/products/{product_id}", response_model=ProductOut)
def update_product(product_id: int, product: ProductIn):
    with LOCK:
        db = load_db()
        for i, p in enumerate(db["products"]):
            if p["id"] == product_id:
                updated = {"id": product_id, **product.dict()}
                db["products"][i] = updated
                save_db(db)
                return updated
    raise HTTPException(status_code=404, detail="Product not found")


@app.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: int):
    with LOCK:
        db = load_db()
        for i, p in enumerate(db["products"]):
            if p["id"] == product_id:
                db["products"].pop(i)
                save_db(db)
                return
    raise HTTPException(status_code=404, detail="Product not found")
