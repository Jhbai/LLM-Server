import config
import asyncio
from model import *
from typing import List, Dict, Any

class Scheduler:
    def __init__(self):
        self.waiting_queue: List[Request] = list() # [Request_obj]
        self.running_map: Dict[str, Request] = {} # {request_id: Request_obj}
        self.lock = asyncio.Lock()

    # ----- Add new task to the scheduler ----- #
    async def add_request(self, request: Request):
        async with self.lock:
            self.waiting_queue += [request]
    
    # ----- Cancel request ----- #
    async def cancel_request(self, request_id: str):
        async with self.lock:
            self.waiting_queue = [r for r in self.waiting_queue if r.request_id != request_id]

            if request_id in self.running_map:
                self.running_map[request_id].cancel()
    
    # ----- Get Next Batch ----- #
    async def get_next_batch(self):
        async with self.lock:
            prefill_batch: List[Request] = list()
            decode_batch: List[Request] = list()

        # ----- Collect all the DECODING request ----- #
        decode_batch = [req for req in self.running_map.values() if req.status == RequestStatus.DECODING]
        decode_batch += [req for req in self.waiting_queue if req.status == RequestStatus.DECODING]
        
        # ----- According to the GPU Resources to allocate the computation results ----- #
        n_available = config.N_CONCURRENCY - len(self.running_map)

        # ----- allocate job when job exist, computing resources remain, be within the limitation of batch ----- #
        while self.waiting_queue and n_available > 0 and len(prefill_batch) < n_available:

            # ----- Let waiting be running ----- #
            req = self.waiting_queue.pop(0)
            req.status = RequestStatus.PREFILLING
            self.running_map[req.request_id] = req
            prefill_batch += [req]

            # ----- counter +1 and -1 ----- #
            n_available -= 1

        return prefill_batch, decode_batch
    
    async def post_process_batch(self, finish_request: List[Request]):
        async with self.lock:
            for req in finish_request:
                if req.request_id in self.running_map:
                    del self.running_map[req.request_id]

        