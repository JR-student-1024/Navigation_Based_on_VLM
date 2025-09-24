import json, torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from nav_vlm.core.types import SceneSummary
from dataclasses import asdict

class QwenNarrator:
    def __init__(self, model_id=r"D:\Python\model\Qwen2.5-VL-3B-Instruct"):
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, device_map="auto", trust_remote_code=True
        )
        self.proc  = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def speak(self, img_path: str, summary: SceneSummary, lang: str="ENG") -> str:
        prompt = f"""
        You are a navigation assistant for the visually impaired.
        Please output in {lang}, limited to two sentences.
        First, state the danger. Then, give a suggestion.
JSON:
{json.dumps(asdict(summary), ensure_ascii=False)}
Format: 
- Warning (using "meter")
- Navigation Instructions (left/right/go straight/wait) """

        conv = [{"role":"user","content":[
            {"type":"image","path":img_path},
            {"type":"text","text":prompt}
        ]}]
        inputs = self.proc.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=96)

        gen = [out[len(x):] for x in inputs.input_ids]
        return self.proc.batch_decode(gen, skip_special_tokens=True)[0]
