from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional

current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from util import create_ort_session, beam_generate_onnx, cut_string_between_bos_eos


os.environ.setdefault("HF_HOME", "/tmp/huggingface")

THIS_DIR = Path(__file__).resolve().parent
DEPLOY_ROOT = THIS_DIR.parent
DEFAULT_MODEL_PATH = DEPLOY_ROOT / "models" / "model-quantized.onnx"
DEFAULT_TOKENIZER_PATH = DEPLOY_ROOT / "models" / "tokenizer"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", str(DEFAULT_TOKENIZER_PATH)))

app = FastAPI(title="Transformer Translation Inference")

tokenizer: Optional[PreTrainedTokenizerBase] = None
ort_session = None
init_error: Optional[str] = None
tokenizer_max_len = int(os.environ.get("TOKENIZER_MAX_LEN", "128"))


class InvocationRequest(BaseModel):
	text: str


def _require_int_token_id(token_id, name: str) -> int:
	if not isinstance(token_id, int):
		raise RuntimeError(f"Tokenizer {name} is not an int.")
	return token_id


def _resolve_path(path: Path, name: str) -> Path:
	if path.exists():
		return path
	raise RuntimeError(f"{name} not found: {path}")


def init_model() -> None:
    global tokenizer, ort_session, init_error
    
    if tokenizer is not None and ort_session is not None:
        return

    try:
        # SageMaker가 S3에서 내려받아 압축을 푼 파일 확인
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        print(f"Loading tokenizer from {TOKENIZER_PATH}...")
        tokenizer_local = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
        
        # 특수 토큰 설정 생략 (기존 로직 유지)
        if tokenizer_local.pad_token is None:
            tokenizer_local.pad_token = tokenizer_local.eos_token or tokenizer_local.cls_token

        print(f"Loading ONNX session from {MODEL_PATH}...")
        session = create_ort_session(MODEL_PATH)

        tokenizer = tokenizer_local
        ort_session = session
        init_error = None
    except Exception as error:
        init_error = str(error)
        print(f"Initialization failed: {init_error}")
        raise

def get_model() -> tuple[PreTrainedTokenizerBase, object]:
	if tokenizer is None or ort_session is None:
		print("Loading model for the first request...")
		init_model()

	if tokenizer is None or ort_session is None:
		raise RuntimeError(init_error or "Model not initialized")

	return tokenizer, ort_session


def _translate_text(text: str) -> str:
	tkn, sess = get_model()

	test_input = tkn(
		[text],
		padding="max_length",
		truncation=True,
		return_tensors="np",
		max_length=tokenizer_max_len,
	)

	input_ids = np.asarray(test_input["input_ids"], dtype=np.int64)
	bos_token_id = _require_int_token_id(tkn.bos_token_id, "bos_token_id")
	eos_token_id = _require_int_token_id(tkn.eos_token_id, "eos_token_id")
	pad_token_id = _require_int_token_id(tkn.pad_token_id, "pad_token_id")

	output_ids = beam_generate_onnx(
		sess,
		src_ids=input_ids,
		bos_token_id=bos_token_id,
		eos_token_id=eos_token_id,
		pad_token_id=pad_token_id,
		max_len=tokenizer_max_len,
		num_beams=3,
		length_penalty=1.0,
		repetition_penalty=1.3,
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

	try:
		translation = _translate_text(request.text)
		return {"translation": translation}
	except HTTPException:
		raise
	except Exception as error:
		raise HTTPException(status_code=500, detail=str(error))


@app.get("/health")
async def health() -> dict:
	return {
		"status": "healthy" if ort_session is not None else "initializing",
		"model_ready": ort_session is not None,
		"error": init_error,
	}

if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
