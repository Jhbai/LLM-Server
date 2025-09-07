import torch
from transformers import AutoProcessor, BitsAndBytesConfig, HybridCache, Gemma3ForConditionalGeneration, DynamicCache
# PATH = "D://LLM//gemma//gemma3_4b"
PATH = "D://LLM//small_gemma//gemma3_270M"

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
model = Gemma3ForConditionalGeneration.from_pretrained(
    PATH,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
    )
model = model.eval()
processor = AutoProcessor.from_pretrained(PATH)