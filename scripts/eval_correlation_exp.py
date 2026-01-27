import os

import csv
import wandb
from collections import Counter
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "You are participating in a simple task. You will see a random sequence of characters. "
    "After seeing it, you must choose exactly one letter: A, B, C, or D. "
    "Just output the single letter, nothing else."
)

USER_TEMPLATE = """Here is a random sequence:
{sequence}
Now choose one option: (A), (B), (C), or (D). Include nothing but the letter of your choosing in your final answer.
Your choice:"""


def load_sequences(filepath: str) -> list:
    """Load sequences from CSV (expects columns: id, sequence)."""
    sequences = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, quotechar='"', doublequote=True)
        for row in reader:
            sequences.append((int(row["id"]), row["sequence"]))
    return sequences


def build_prompt(tok, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_choice(response: str) -> str:
    """Extract A, B, C, or D from response."""
    response = response.strip().upper()
    # Remove parentheses if present
    response = response.replace("(", "").replace(")", "")
    if response and response[0] in "ABCD":
        return response[0]
    for char in response:
        if char in "ABCD":
            return char
    return "X"


def run_model(llm: LLM, tok, cfg, items: list):
    """Run model on all sequences, return wandb table."""
    prompts = [build_prompt(tok, cfg.system, USER_TEMPLATE.format(sequence=seq)) 
               for _, seq in items]
    
    sp = SamplingParams(
        n=1,
        temperature=float(cfg.temperature),
        max_tokens=int(cfg.max_tokens),
    )
    outs = llm.generate(prompts, sp)

    table = wandb.Table(columns=["id", "sequence", "extracted_choice", "response"])
    choices = []
    for (sid, seq), out in zip(items, outs):
        txt = out.outputs[0].text.strip()
        choice = extract_choice(txt)
        choices.append(choice)
        table.add_data(sid, seq, choice, txt)
    
    dist = Counter(choices)
    print(f"Choice distribution: {dict(dist)}")
    
    return table


def main():
    defaults = dict(
        model="Qwen/Qwen3-32B",
        sequences_path="correlation_data/correlation_data.csv",
        temperature=0.0,
        max_tokens=10,
        seed=0,
        dtype="half",
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        system=SYSTEM_PROMPT,
    )

    run = wandb.init(config=defaults)
    cfg = wandb.config

    items = load_sequences(str(cfg.sequences_path))
    print(f"Loaded {len(items)} sequences")

    tok = AutoTokenizer.from_pretrained(str(cfg.model), trust_remote_code=True)
    llm = LLM(
        model=str(cfg.model),
        trust_remote_code=True,
        dtype=str(cfg.dtype),
        tensor_parallel_size=int(cfg.tensor_parallel_size),
        max_model_len=int(cfg.max_model_len),
        gpu_memory_utilization=float(cfg.gpu_memory_utilization),
    )

    table = run_model(llm, tok, cfg, items)
    wandb.log({"samples": table})

    run.finish()


if __name__ == "__main__":
    main()