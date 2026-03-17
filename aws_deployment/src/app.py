import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase

current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from util import load_model, beam_generate, cut_string_between_bos_eos


os.environ.setdefault("HF_HOME", "/tmp/huggingface")

THIS_DIR = Path(__file__).resolve().parent
DEPLOY_ROOT = THIS_DIR.parent
DEFAULT_MODEL_PATH = DEPLOY_ROOT / "models" / "model.pt"
DEFAULT_MODEL_CONFIG_PATH = DEPLOY_ROOT / "models" / "model-config.json"
DEFAULT_TOKENIZER_PATH = DEPLOY_ROOT / "models" / "tokenizer"

MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
MODEL_CONFIG_PATH = Path(os.environ.get("MODEL_CONFIG_PATH", str(DEFAULT_MODEL_CONFIG_PATH)))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", str(DEFAULT_TOKENIZER_PATH)))

app = FastAPI(title="Transformer Translation Inference")

tokenizer: Optional[PreTrainedTokenizerBase] = None
pt_model = None
init_error: Optional[str] = None
tokenizer_max_len = int(os.environ.get("TOKENIZER_MAX_LEN", "128"))


class InvocationRequest(BaseModel):
	text: str
	temperature: float = Field(default=0.9, gt=0.0, le=5.0)
	top_k: int = Field(default=50, ge=0, le=1000)
	top_p: float = Field(default=0.95, gt=0.0, le=1.0)
	repetition_penalty: float = Field(default=1.2, ge=1.0, le=3.0)
	num_beams: int = Field(default=3, ge=1, le=10)
	length_penalty: float = Field(default=1.0, ge=0.1, le=3.0)
	seed: Optional[int] = Field(default=None, ge=0)


def init_model() -> None:
    global tokenizer, pt_model, init_error

    if tokenizer is not None and pt_model is not None:
        return

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not MODEL_CONFIG_PATH.exists():
            raise FileNotFoundError(f"Model config not found at {MODEL_CONFIG_PATH}")

        print(f"Loading tokenizer from {TOKENIZER_PATH}...")
        tokenizer_local = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
        if tokenizer_local.pad_token is None:
            tokenizer_local.pad_token = tokenizer_local.eos_token or tokenizer_local.cls_token

        with open(MODEL_CONFIG_PATH) as f:
            config = json.load(f)

        print(f"Loading PyTorch model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH, config)
        model.tokenizer = tokenizer_local

        tokenizer = tokenizer_local
        pt_model = model
        init_error = None
        print("Model ready.")
    except Exception as error:
        init_error = str(error)
        print(f"Initialization failed: {init_error}")
        raise


def get_model():
    if tokenizer is None or pt_model is None:
        print("Loading model for the first request...")
        init_model()

    if tokenizer is None or pt_model is None:
        raise RuntimeError(init_error or "Model not initialized")

    return tokenizer, pt_model


def _translate_text(
	text: str,
	*,
	repetition_penalty: float,
	num_beams: int,
	length_penalty: float,
) -> str:
	tkn, model = get_model()

	encoding = tkn(
		[text],
		padding="max_length",
		truncation=True,
		return_tensors="pt",
		max_length=tokenizer_max_len,
	)
	input_ids: torch.Tensor = encoding.input_ids

	output_ids = beam_generate(
		model,
		src_ids=input_ids,
		num_beams=num_beams,
		max_len=tokenizer_max_len,
		length_penalty=length_penalty,
		repetition_penalty=repetition_penalty,
	)

	output_sentences = tkn.batch_decode(output_ids.tolist(), skip_special_tokens=False)
	return cut_string_between_bos_eos(output_sentences[0])


@app.get("/ping")
async def ping() -> Response:
	try:
		get_model()
		return Response(status_code=200)
	except Exception as error:
		return Response(content=str(error), status_code=503)


@app.post("/invocations")
async def invocations(request: InvocationRequest) -> dict:
	if not request.text.strip():
		raise HTTPException(status_code=400, detail="Empty input text")

	print(f"DEBUG: Processing Beam Search request. num_beams={request.num_beams}, temp={request.temperature}")

	try:
		translation = _translate_text(
			request.text,
			repetition_penalty=request.repetition_penalty,
			num_beams=request.num_beams,
			length_penalty=request.length_penalty,
		)
		return {"translation": translation}
	except HTTPException:
		raise
	except Exception as error:
		raise HTTPException(status_code=500, detail=str(error))


@app.get("/health")
async def health() -> dict:
	return {
		"status": "healthy" if pt_model is not None else "initializing",
		"model_ready": pt_model is not None,
		"error": init_error,
	}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
