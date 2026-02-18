from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import tempfile

app = FastAPI()
@app.get("/healthz")
def health():
    return {"status": "ok"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading AI model... (first time takes 1–2 mins)")
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)
model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
)

@app.post("/ask-image")
async def ask_image(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await image.read())
    temp.close()

    img = Image.open(temp.name).convert("RGB")

    inputs = processor(img, question, return_tensors="pt")
    out = model.generate(**inputs)

    answer = processor.decode(out[0], skip_special_tokens=True)

    return {"answer": answer}



