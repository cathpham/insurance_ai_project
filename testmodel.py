from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Optimize for memory
    device_map="auto"  # Use GPU (Metal) if available, else CPU
)

prompt = "Hello, explain why insurance costs might be high for a smoker."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))