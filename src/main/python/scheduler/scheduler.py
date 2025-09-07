import gc
import torch
from typing import List, Dict, Any
from collections import OrderedDict
from src.main.python.schema import model 
from src.main.python.config import config
from src.main.python.engine import cache_manager
from src.main.python.engine import decode, prefill
from transformers import DynamicCache, Gemma3ForConditionalGeneration

class RequestManager:
    def __init__(self):
        self.RequestsList: Dict[model.Request.request_id, model.Request] = OrderedDict()

    def add_request(self, request: model.Request):
        self.RequestsList[request.request_id] = request

    def step(self):
        
        for batch in range(config.BATCH_SIZE):
