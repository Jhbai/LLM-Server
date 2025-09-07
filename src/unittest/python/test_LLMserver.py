import uuid
import torch
import unittest
from transformers import DynamicCache
from src.main.python.schema import model

class TestCalculations(unittest.TestCase):
    def test_schema(self):
        request = model.Request(input_ids = torch.tensor([i for i in range(128)]).unsqueeze(0),
                                status = model.RequestStatus.PREFILLING,
                                request_id = uuid.uuid4(),
                                kv_cache = DynamicCache()
        ) 
        # ----- 測試 get_ids 是不是只拿一個 batch 量的 seq_len ----- #
        vals = request.get_ids()
        test_get_ids = (vals == torch.tensor([[i for i in range(16)]])).all()
        self.assertEqual(test_get_ids, True)

        # ----- 測試 tidx 是不是有一起更新 ----- #
        tidx = request.tidx
        test_tidx = (tidx == 16)
        self.assertEqual(test_tidx, True)

        # ----- 測試 δ(PREFILLING, Prefilling_Complete) 的轉換 ----- #
        request = model.Request(input_ids = torch.tensor([i for i in range(128)]).unsqueeze(0),
                                status = model.RequestStatus.PREFILLING,
                                request_id = uuid.uuid4(),
                                kv_cache = DynamicCache()
        ) 
        for _ in range(8):
            request.get_ids()
        vals = request.get_ids()
        test_final_idx = (int(vals) == 127)
        self.assertEqual(test_final_idx, True)
        test_status = (request.status is model.RequestStatus.DECODING)
        self.assertEqual(test_status, True) 