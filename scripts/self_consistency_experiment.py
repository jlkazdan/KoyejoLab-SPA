# scripts/self_consistency_experiment.py
import os
import re
import wandb
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYS_DEFAULT = (
    "You are a careful math assistant. Output ONLY the final answer in LaTeX as "
    "\\boxed{...} with no extra text."
)
MATH_CFGS = "algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus".split()


def boxed(s: str) -> str:
    """Extract the *content* of the last \\boxed{...} in string s. Returns '' if not found."""
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


def _to_int_year(x) -> int:
    s = str(x).strip().replace(",", "")
    return int(s)


def _load_split_with_fallback(path: str, preferred: str):
    for sp in [preferred, "test", "validation", "train"]:
        try:
            return load_dataset(path, split=sp)
        except Exception:
            pass
    raise ValueError(f"Could not load {path} with split={preferred} (or fallbacks).")


def expand_datasets(datasets_field):
    """
    Accepts either:
      - list/tuple: ["svamp","gsm8k","math","aime"]
      - string: "svamp,gsm8k,math,aime"
    """
    if isinstance(datasets_field, (list, tuple)):
        toks = [str(x).strip() for x in datasets_field if str(x).strip()]
    else:
        toks = [x.strip() for x in str(datasets_field).split(",") if x.strip()]
    return toks


def load_items_for_dataset(cfg, name: str):
    """
    Returns list of tuples (dataset, qid, question, gold)
    IMPORTANT: this returns items for ONE dataset only (no mixing).
    """
    seed, n = int(cfg.seed), int(cfg.max_samples)

    if name == "gsm8k":
        d = load_dataset("openai/gsm8k", "main", split=str(cfg.split))
        d = d.shuffle(seed=seed).select(range(min(n, len(d))))
        items = []
        for qid, ex in enumerate(d):
            q = ex["question"]
            g = ex["answer"].split("####")[-1].strip()
            items.append((name, qid, q, g))
        return items

    if name == "svamp":
        d = load_dataset("ChilleD/SVAMP", split=str(cfg.split))
        d = d.shuffle(seed=seed).select(range(min(n, len(d))))
        items = []
        for qid, ex in enumerate(d):
            q = ex.get("question_concat") or (ex["Body"] + " " + ex["Question"])
            g = str(ex["Answer"]).strip()
            items.append((name, qid, q, g))
        return items

    if name == "math":
        d = concatenate_datasets(
            [load_dataset("EleutherAI/hendrycks_math", c, split=str(cfg.split)) for c in MATH_CFGS]
        )
        d = d.shuffle(seed=seed).select(range(min(n, len(d))))
        items = []
        for qid, ex in enumerate(d):
            q = ex["problem"]
            g = (boxed(ex["solution"]) or ex["solution"]).strip()
            items.append((name, qid, q, g))
        return items

    # Simple AIME: sample from gneubig/aime-1983-2024, default <= 2024.
    # Optional wandb overrides:
    #   aime_min_year: int (default 0)
    #   aime_max_year: int (default 2024)
    if name == "aime":
        y_min = int(getattr(cfg, "aime_min_year", 0))
        y_max = int(getattr(cfg, "aime_max_year", 2024))

        d = _load_split_with_fallback("gneubig/aime-1983-2024", preferred=str(cfg.split))
        d = d.filter(lambda ex: y_min <= _to_int_year(ex["Year"]) <= y_max)
        d = d.shuffle(seed=seed).select(range(min(n, len(d))))

        items = []
        for qid, ex in enumerate(d):
            q = ex.get("Question") or ex.get("Problem") or ex.get("problem")
            g = str(ex.get("Answer")).strip()

            yr = _to_int_year(ex.get("Year"))
            part = str(ex.get("Part", "")).strip()
            pn = ex.get("Problem Number", ex.get("Problem_Number", ""))

            qkey = ex.get("ID", None)
            if qkey is None:
                pn_s = str(pn).strip()
                qkey = f"{yr}{part}-{pn_s}" if pn_s else f"{yr}{part}-{qid}"

            items.append((name, qkey, q, g))
        return items

    raise ValueError(f"unknown dataset token: {name}")


def build_prompt(tok, system: str, q: str) -> str:
    msgs = [{"role": "system", "content": str(system)}, {"role": "user", "content": q}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{system}\n\n{q}\n\nAnswer: "


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
        gbox = f"\\boxed{{{g}}}"
        for sid, o in enumerate(out.outputs):
            txt = o.text.strip()
            p = boxed(txt)
            table.add_data(ds_name, qid, sid, q, gbox, (f"\\boxed{{{p}}}" if p else ""), txt)

    return table


def main():
    # Defaults (sweep overrides these)
    defaults = dict(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        datasets=["svamp", "gsm8k", "math"],
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
        # AIME knobs (only used if datasets includes "aime")
        aime_min_year=0,
        aime_max_year=2024,
        # Optional: set via wandb if you need it
        # vllm_attention_backend="TRITON_ATTN_VLLM_V1",
    )

    run = wandb.init(config=defaults)
    cfg = wandb.config

    # Optional vLLM env knobs from wandb config
    if getattr(cfg, "vllm_attention_backend", None):
        os.environ["VLLM_ATTENTION_BACKEND"] = str(cfg.vllm_attention_backend)

    tok = AutoTokenizer.from_pretrained(str(cfg.model), trust_remote_code=True)
    llm = LLM(
        model=str(cfg.model),
        trust_remote_code=True,
        dtype=str(cfg.dtype),
        tensor_parallel_size=int(cfg.tensor_parallel_size),
        max_model_len=int(cfg.max_model_len),
        gpu_memory_utilization=float(cfg.gpu_memory_utilization),
    )

    ds_list = expand_datasets(cfg.datasets)

    first = True
    for ds in ds_list:
        table = run_one_dataset(llm, tok, cfg, ds)

        # Log per-dataset table under its own key (no mixing).
        wandb.log({f"samples/{ds}": table})

        # Also log ONE table to key "samples" to satisfy sweep schema.
        if first:
            wandb.log({"samples": table})
            first = False

    run.finish()


if __name__ == "__main__":
    main()
