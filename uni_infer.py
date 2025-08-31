import torch
from transformers import AutoProcessor, BitsAndBytesConfig, HybridCache, Gemma3ForConditionalGeneration, DynamicCache
from PIL import Image, ImageGrab
# PATH = "./gemma/gemma3_4b"
PATH = "D://LLM//gemma//gemma3_4b"

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

def gemma3_resp(prompt, image=None):
    max_seq_len = 10000

    # ----- 結果儲存 ----- #
    res = list()

    # ----- Prompt token產生 ----- #
    eos_token_ids = [processor.tokenizer.eos_token_id, 106]
    if image is not None:
        msg = "<start_of_turn>user\n<start_of_image>{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
        image = image.convert("RGB")
        MSG = msg.format(prompt=prompt)
        inputs = processor(text=MSG, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        pixel_values = inputs["pixel_values"].to(model.device)
        
    else:
        msg = """<start_of_turn>user\n{prompt}\n<end_of_turn><start_of_turn>model\n"""
        MSG = msg.format(prompt=prompt)
        inputs = processor(text=MSG, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

    # ----- Cache宣告 ----- #
    past_key_values = DynamicCache()
    
    # ----- Prefill ----- #
    chunks = torch.split(input_ids[:, :-1], 300, dim=-1)
    st = 0
    ed = 0
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            ed = st + chunk.shape[1]
            if i == 0 and image is not None:
                model(input_ids=chunk,
                    pixel_values=pixel_values,
                    use_cache=True, 
                    past_key_values=past_key_values)
            else:
                model(input_ids=chunk,
                    use_cache=True, 
                    past_key_values=past_key_values)
            st = ed
    
    # ----- Auto Regressive生成 ----- #
    input_ids = input_ids[:, -1:]
    attention_mask = torch.ones(1, ed, dtype=torch.long, device=model.device)
    try:
        for _ in range(max_seq_len):
            with torch.no_grad():
                # ----- Update position ----- #
                ed += 1

                # ----- Update model kwargs ----- #
                cache_position = torch.arange(ed-1, ed, dtype=torch.long, device = model.device)

                # ----- 生成token ----- #
                outputs = model(input_ids=input_ids, 
                                use_cache=True, 
                                past_key_values=past_key_values, 
                                cache_position=cache_position)
                logits = outputs.logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                token_id = next_token.item()
                input_ids = next_token

                # ----- 判斷是否終止 ----- #
                if token_id in eos_token_ids:
                    break

                # ----- 紀錄token ----- #
                res += [processor.decode(token_id)]
                
                # ----- 輸出文字字串 ----- #
                print(res[-1], end="", flush=True)
    except:
        for item in ("input_ids", "outputs", "ogits", "next_token", "token_id"):
            try:
                eval(f"del {item}")
            except:
                pass
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    return "".join(res)