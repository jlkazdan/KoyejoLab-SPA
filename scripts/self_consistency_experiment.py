# scripts/self_consistency_experiment.py
import os
import re
import wandb
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYS_DEFAULT = (
    "You are a careful math assistant. After solving the problem, output the final answer in LaTeX as "
    "\\boxed{...} with no extra text inside of the box."
)
MATH_CFGS = "algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus".split()


def boxed(s: str) -> str:
    i = s.rfind(r"\boxed")
    if i < 0:
        return ""
    j = s.find("{", i)
    if j < 0:
        return ""
    d = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0:
                return s[j + 1 : k].strip()
    return ""


def expand_datasets(datasets_field):
    if isinstance(datasets_field, (list, tuple)):
        return [str(x).strip() for x in datasets_field if str(x).strip()]
    return [x.strip() for x in str(datasets_field).split(",") if x.strip()]


def load_items_for_dataset(cfg, name: str):
    seed, n = int(cfg.seed), int(cfg.max_samples)

    if name == "gsm8k":
        d = load_dataset("openai/gsm8k", "main", split=str(cfg.split))
        d = d.shuffle(seed=seed).select(range(min(n, len(d))))
        return [(name, i, ex["question"], ex["answer"].split("####")[-1].strip())
                for i, ex in enumerate(d)]

    if name == "svamp":
        d = load_dataset("ChilleD/SVAMP", split=str(cfg.split))
        d = d.shuffle(seed=seed).select(range(min(n, len(d))))
        return [(name, i,
                 ex.get("question_concat") or (ex["Body"] + " " + ex["Question"]),
                 str(ex["Answer"]).strip())
                for i, ex in enumerate(d)]

    if name == "math":
        d = concatenate_datasets(
            [load_dataset("EleutherAI/hendrycks_math", c, split=str(cfg.split)) for c in MATH_CFGS]
        )
        d = d.shuffle(seed=seed).select(range(min(n, len(d))))
        return [(name, i, ex["problem"],
                 boxed(ex["solution"]) or ex["solution"])
                for i, ex in enumerate(d)]

    if name == "aime":
        d = load_dataset("gneubig/aime-1983-2024", split=str(cfg.split))
        d = d.shuffle(seed=seed).select(range(min(n, len(d))))
        return [(name, i, ex["Question"], str(ex["Answer"]).strip())
                for i, ex in enumerate(d)]

    raise ValueError(f"unknown dataset token: {name}")


def build_prompt(tok, system: str, q: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run_one_dataset(llm: LLM, tok, cfg, ds: str):
    items = load_items_for_dataset(cfg, ds)

    prompts = [build_prompt(tok, cfg.system, q) for _, _, q, _ in items]
    sp = SamplingParams(
        n=int(cfg.k),
        temperature=float(cfg.temperature),
        top_p=float(cfg.top_p),
        max_tokens=int(cfg.max_new_tokens),
    )
    outs = llm.generate(prompts, sp)

    table = wandb.Table(columns=["dataset", "qid", "sample_id", "question", "gold", "pred", "response"])
    for (ds_name, qid, q, g), out in zip(items, outs):
        for sid, o in enumerate(out.outputs):
            txt = o.text.strip()
            p = boxed(txt)
            table.add_data(ds_name, qid, sid, q, f"\\boxed{{{g}}}",
                           f"\\boxed{{{p}}}" if p else "", txt)
    return table


def main():
    defaults = dict(
        model="Qwen/Qwen3-32B",
        datasets=["math", "aime"],
        split="test",
        max_samples=10,
        k=200,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=1500,
        seed=0,
        dtype="half",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        system=SYS_DEFAULT,
    )

    run = wandb.init(config=defaults)
    cfg = wandb.config

    tok = AutoTokenizer.from_pretrained(str(cfg.model), trust_remote_code=True)
    llm = LLM(
        model=str(cfg.model),   # ← FIXED HERE
        trust_remote_code=True,
        dtype=str(cfg.dtype),
        tensor_parallel_size=int(cfg.tensor_parallel_size),
        max_model_len=int(cfg.max_model_len),
        gpu_memory_utilization=float(cfg.gpu_memory_utilization),
    )

    first = True
    for ds in expand_datasets(cfg.datasets):
        table = run_one_dataset(llm, tok, cfg, ds)
        wandb.log({f"samples/{ds}": table})
        if first:
            wandb.log({"samples": table})
            first = False

    run.finish()


if __name__ == "__main__":
    main()
