from transformers import AutoTokenizer

model_name = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model: {model_name}")
print(f"BOS token: {tokenizer.bos_token}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"SEP token: {tokenizer.sep_token}")
print(f"CLS token: {tokenizer.cls_token}")
print(f"PAD token: {tokenizer.pad_token}")
print(f"MASK token: {tokenizer.mask_token}")
