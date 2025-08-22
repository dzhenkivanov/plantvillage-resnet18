from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.inference import load_model, predict_image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model, class_names, transform = load_model()

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "File must be an image")
    img_bytes = await file.read()
    topk = predict_image(img_bytes, model, transform, class_names, k=3)
    return {"topk": topk}
