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
{prompt}<end_of_turn>
<start_of_turn>model
"""

# Event = multiprocessing.Event()
# manager = multiprocessing.Manager()
# CacheDict = manager.dict()
# TaskQueue = manager.Queue()
# ResDict = manager.dict()

# ----- 分詞器建立在API上 ----- #
PATH = "C://Users//user//LLM//smallgemma3"
tokenizer = GemmaTokenizerFast.from_pretrained(PATH)

# ----- Process函數+載入模型 ----- #
def Inference_Engine(TaskQueue, ResDict):
    _scheduler = scheduler.RequestManager()
    CacheDict = dict() # Cache本質不應該被傳遞
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
        if not TaskQueue.empty():
            ids, request_id, status = TaskQueue.get_nowait()
            Cache = CacheDict.get(request_id, DynamicCache())
            ids_tensor = torch.tensor(ids)
            if len(ids_tensor.shape) < 2:
                ids_tensor = ids_tensor.unsqueeze(0)
            print("TaskQueue Input:", ids_tensor)
            request = model.Request(
                                input_ids = ids_tensor,
                                status = status,
                                request_id = request_id,
                                kv_cache = Cache
            )
            _scheduler.add_request(request)
        d_input_ids, d_caches, p_input_ids, p_caches, decode_requests = _scheduler.step()
        UUID = [r.request_id for r in decode_requests]
        TEXT, DCACHES = decode.infer(llm_model, d_input_ids, d_caches, UUID)
        _, PCACHES = prefill.infer(llm_model, p_input_ids, p_caches)
        _scheduler.update(PCACHES)

        if TEXT is not None:
            for i, uid in enumerate(UUID):
                ResDict[uid] = TEXT[uid]
                CacheDict[uid] = DCACHES[i]
                
async def RequestsHandler(uid, TaskQueue, ResDict):
    end_flag = False
    text = ""
    while "<end_of_turn>" not in text:
        if uid in ResDict:
            ids = ResDict.pop(uid)
        else:
            continue
        if 106 in ids:
            end_flag = True
        input_ids = torch.tensor(ids[-1:]).unsqueeze(0)
        TaskQueue.put((input_ids, uid, model.RequestStatus.DECODING))
        text += tokenizer.decode(ids, skip_special_tokens=True)
        yield tokenizer.decode(ids, skip_special_tokens=True)
        if end_flag:
            break
    
# ----- 建立API服務 ----- #
app = FastAPI(title="LLM Service", version="2.0.0")
@app.on_event("startup")
def startup():
    manager = multiprocessing.Manager()
    app.state.TaskQueue = manager.Queue()
    app.state.ResDict = manager.dict()

    app.state.worker = multiprocessing.Process(
        target=Inference_Engine,
        args=(app.state.TaskQueue, app.state.ResDict),
        daemon=True # Daemon means that if the parent process've been terminated, then child will be terminated as well.
    )
    app.state.worker.start()

@app.post("/v1/chat/completions")
async def LLM_Response(uid: str, prompt: str):
    TaskQueue, ResDict = app.state.TaskQueue, app.state.ResDict
    ids = tokenizer.encode(MSG.format(prompt=prompt))
    TaskQueue.put((ids, uid, model.RequestStatus.PREFILLING))
    return StreamingResponse(RequestsHandler(uid, TaskQueue, ResDict), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, workers=2, reload=False)