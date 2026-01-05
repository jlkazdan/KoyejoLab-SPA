# eval_k_math_vllm.py
import wandb
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYS = "You are a careful math assistant. Output ONLY the final answer in LaTeX as \\boxed{...} with no extra text."
MATH_CFGS = "algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus".split()

def boxed(s: str) -> str:
    i = s.rfind(r"\boxed")
    if i < 0: return ""
    j = s.find("{", i)
    if j < 0: return ""
    d = 0
    for k in range(j, len(s)):
        if s[k] == "{": d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0: return s[j+1:k].strip()
    return ""

def load_items(cfg):
    items, seed, n = [], int(cfg.seed), int(cfg.max_samples)
    for name in [x.strip() for x in str(cfg.datasets).split(",") if x.strip()]:
        if name == "gsm8k": d = load_dataset("openai/gsm8k", "main", split=cfg.split)
        elif name == "svamp": d = load_dataset("ChilleD/SVAMP", split=cfg.split)
        elif name == "math": d = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", c, split=cfg.split) for c in MATH_CFGS])
        else: raise ValueError(f"unknown dataset: {name}")
        d = d.shuffle(seed=seed).select(range(min(n, len(d))))
        for qid, ex in enumerate(d):
            if name == "gsm8k":
                q, g = ex["question"], ex["answer"].split("####")[-1].strip()
            elif name == "svamp":
                q = ex.get("question_concat") or (ex["Body"] + " " + ex["Question"])
                g = str(ex["Answer"]).strip()
            else:
                q, g = ex["problem"], (boxed(ex["solution"]) or ex["solution"]).strip()
            items.append((name, qid, q, g))
    return items

def main():
    defaults = dict(model="meta-llama/Meta-Llama-3-8B-Instruct", datasets="gsm8k,math,svamp", split="test",
                    max_samples=200, k=200, temperature=0.7, top_p=0.95, max_new_tokens=128, seed=0, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.9,
                    system=SYS)
    run = wandb.init(config=defaults); cfg = wandb.config

    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    llm = LLM(model=cfg.model, trust_remote_code=True, dtype=str(cfg.dtype),
              tensor_parallel_size=int(cfg.tensor_parallel_size),
              max_model_len=int(cfg.max_model_len),
              gpu_memory_utilization=float(cfg.gpu_memory_utilization))

    items = load_items(cfg)
    prompts = []
    for _, _, q, _ in items:
        msgs = [{"role":"system","content":str(cfg.system)}, {"role":"user","content":q}]
        try: prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception: prompts.append(f"{cfg.system}\n\n{q}\n\nAnswer: ")

    sp = SamplingParams(n=int(cfg.k), temperature=float(cfg.temperature), top_p=float(cfg.top_p),
                        max_tokens=int(cfg.max_new_tokens))
    outs = llm.generate(prompts, sp)

    table = wandb.Table(columns=["dataset","qid","sample_id","question","gold","pred","response"])
    for (ds, qid, q, g), out in zip(items, outs):
        gbox = f"\\boxed{{{g}}}"
        for sid, o in enumerate(out.outputs):
            txt = o.text.strip()
            p = boxed(txt)
            table.add_data(ds, qid, sid, q, gbox, (f"\\boxed{{{p}}}" if p else ""), txt)

    wandb.log({"samples": table})
    run.finish()

if __name__ == "__main__":
    main()
