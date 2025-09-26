"This script has been adapted from `https://github.com/QwenLM/Qwen3Guard` and `https://modal.com/docs/examples/vllm_inference`"

import modal
import re

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "fastapi", "transformers", "torch", "hf_transfer", "openai", "accelerate"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)  # faster model transfers
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


app = modal.App("Qwen3Guard", image=image)


def extract_label_and_categories(content):
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
    safe_label_match = re.search(safe_pattern, content)
    label = safe_label_match.group(1) if safe_label_match else None
    categories = re.findall(category_pattern, content)
    return label, categories


@app.cls(
    gpu="A10G",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
class Qwen3GuardRouter:
    MODEL_NAME = "Qwen/Qwen3Guard-Gen-4B"

    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def predict(self, messages: list[dict]) -> str:
        import torch
        from datetime import datetime

        start_time = datetime.now()
        assert torch.cuda.is_available()

        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=128)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return content, datetime.now() - start_time
