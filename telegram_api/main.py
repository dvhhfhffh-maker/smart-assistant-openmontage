from fastapi import FastAPI

app = FastAPI(
    title="Smart Assistant OpenMontage API",
    version="0.1.0"
)

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Smart Assistant OpenMontage API"
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "status": "healthy"
    }
