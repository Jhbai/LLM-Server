import os
import io
import sys
import uuid
import time
import json
import torch
import queue
import asyncio
import uvicorn
import importlib
import threading
import subprocess
import multiprocessing
from fastapi import FastAPI, APIRouter
from fastapi.responses import StreamingResponse
from transformers import GemmaTokenizerFast, BitsAndBytesConfig, Gemma3ForCausalLM, DynamicCache

from src.main.python.schema import model
from src.main.python.config import config
from src.main.python.scheduler import scheduler
from src.main.python.engine import decode, prefill
# ----- 初始化Server ----- #
TEXT = dict()
MSG = """<start_of_turn>user
[所有回應一律用繁體中文回答]{prompt}<end_of_turn>
<start_of_turn>model
"""
Event = multiprocessing.Event()
manager = multiprocessing.Manager()
CacheDict = manager.dict()
TaskQueue = manager.Queue()
ResDict = manager.dict()

# ----- 分詞器建立在API上 ----- #
PATH = "D://LLM//gemma//gemma3_4b"
tokenizer = GemmaTokenizerFast.from_pretrained(PATH)

# ----- Process函數+載入模型 ----- #
def Inference_Engine():
    _scheduler = scheduler.RequestManager()
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

    llm_model = Gemma3ForCausalLM.from_pretrained(
        PATH,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    llm_model = llm_model.eval()

    while True:
        try:
            ids, request_id = TaskQueue.get_nowait()
            Cache = CacheDict.get(request_id, DynamicCache())
            request = model.Request(
                                input_ids = torch.tensor(ids).unsqueeze(0),
                                status = model.RequestStatus.PREFILLING,
                                request_id = request_id,
                                kv_cache = Cache
            )
            _scheduler.add_request(request)
        except TaskQueue.Empty:
            pass
        d_input_ids, d_caches, p_input_ids, p_caches, decode_requests = _scheduler.step()
        UUID = [r.request_id for r in decode_requests]
        TEXT, DCACHES = decode.infer(llm_model, d_input_ids, d_caches, UUID)
        _, PCACHES = prefill.infer(llm_model, p_input_ids, p_caches)
        _scheduler.update(PCACHES)

        if TEXT is not None:
            for i, uid in enumerate(UUID):
                ResDict[uid] = (TEXT[uid], DCACHES[i])
                
async def RequestsHandler(uid):
    text = ""
    while "<end_of_turn>" not in text:
        ids, cache = ResDict[uid]
        CacheDict[uid] = cache
        input_ids = torch.tensor(ids[-1:]).unsqueeze(0)
        TaskQueue.put(input_ids, uid)
        text += tokenizer.decode(ids, skip_special_tokens=True)
        yield text
    CacheDict[uid] = cache
    
# ----- 建立API服務 ----- #
router = APIRouter(prefix="/v1", tags=["LLM Inference"])
@router.post("/chat/completions")
async def LLM_Response(uid: str, prompt: str):
    if uid not in CacheDict:
        uid = str(uuid.uuid4())
    Cache = CacheDict.get(uid, DynamicCache())
    ids = tokenizer.encode(MSG.format(prompt=prompt))
    TaskQueue.put(ids, uid)
    return StreamingResponse(RequestsHandler(uid), media_type="text/plain")

app = FastAPI(title="LLM Service", version="2.0.0")
app.include_router(router)

if __name__ == "__main__":
    prod_process = multiprocessing.Process(
            target=Inference_Engine, 
            name="LLM_Inference_Engine"
        )
    uvicorn.run("main:app", host="127.0.0.1", port=8000, workers=8, reload=True)