#!/usr/bin/env python3
"""
Temperature Stability Analysis: Measure how often majority answers flip between T=0.7 and T=1.0
"""

import pandas as pd
import numpy as np
from collections import Counter

# Load the response data from BOTH files
print("Loading data...")
df1 = pd.read_csv('data/136b2d5c41279e432cafd46eb4939875_responses.csv')
df2 = pd.read_csv('data/c4c73e31a5f002c6df19d36e7aebb3c6_responses.csv')
df = pd.concat([df1, df2], ignore_index=True)
print(f"Total rows: {len(df):,} (from 2 CSV files)")

# Filter to direct_answer responses only (these contain the actual answers)
df_answers = df[df['response_type'] == 'direct_answer'].copy()
print(f"Direct answer rows: {len(df_answers):,}")

# Clean dataset names for readability
dataset_names = {
    'cais/hle': 'HLE',
    'google/boolq': 'BoolQ',
    'tasksource/com2sense': 'Com2Sense',
    'kyssen/predict-the-futurebench-cutoff-June25': 'Predict-the-Future'
}
df_answers['benchmark'] = df_answers['config_dataset_name'].map(dataset_names)

# Clean model names for readability
model_names = {
    'google/gemma-3-4b-it': 'Gemma-3-4B',
    'openai.gpt-oss-120b-1:0': 'GPT-OSS-120B',
    'openai.gpt-oss-20b-1:0': 'GPT-OSS-20B',
    'qwen.qwen3-235b-a22b-2507-v1:0': 'Qwen3-235B',
    'qwen.qwen3-32b-v1:0': 'Qwen3-32B'
}
df_answers['model'] = df_answers['config_model_id'].map(model_names)

def get_majority_answer(answers):
    """Return the majority answer from a list of answers."""
    # Filter out NaN/None values
    valid_answers = [a for a in answers if pd.notna(a)]
    if not valid_answers:
        return None
    counter = Counter(valid_answers)
    return counter.most_common(1)[0][0]

def compute_flip_rates(df_answers):
    """Compute flip rates for each (benchmark, model, question) tuple."""
    results = []

    # Group by benchmark, model, question
    grouped = df_answers.groupby(['benchmark', 'model', 'question_idx'])

    for (benchmark, model, question_idx), group in grouped:
        # Get answers at each temperature
        t07 = group[group['config_temperature'] == 0.7]['extracted_answer'].tolist()
        t10 = group[group['config_temperature'] == 1.0]['extracted_answer'].tolist()

        # Skip if we don't have both temperatures
        if not t07 or not t10:
            continue

        # Get majority answers
        majority_07 = get_majority_answer(t07)
        majority_10 = get_majority_answer(t10)

        # Record result
        flipped = majority_07 != majority_10
        results.append({
            'benchmark': benchmark,
            'model': model,
            'question_idx': question_idx,
            'majority_07': majority_07,
            'majority_10': majority_10,
            'flipped': flipped,
            'n_samples_07': len(t07),
            'n_samples_10': len(t10)
        })

    return pd.DataFrame(results)

print("\nComputing flip rates...")
flip_df = compute_flip_rates(df_answers)
print(f"Total question comparisons: {len(flip_df):,}")

# Verify sample counts
print(f"\nSample counts at T=0.7: {flip_df['n_samples_07'].value_counts().to_dict()}")
print(f"Sample counts at T=1.0: {flip_df['n_samples_10'].value_counts().to_dict()}")

# Overall flip rate
overall_flip_rate = flip_df['flipped'].mean() * 100
print(f"\n{'='*60}")
print(f"OVERALL FLIP RATE: {overall_flip_rate:.1f}%")
print(f"{'='*60}")

# Report unique questions per benchmark
print("\n--- Unique Questions per Benchmark ---")
for bench in sorted(flip_df['benchmark'].unique()):
    n_unique = flip_df[flip_df['benchmark'] == bench]['question_idx'].nunique()
    print(f"  {bench}: {n_unique} questions")

# Flip rate by benchmark
print("\n--- Flip Rate by Benchmark ---")
benchmark_flip = flip_df.groupby('benchmark').agg(
    flip_rate=('flipped', 'mean'),
    n_comparisons=('flipped', 'count'),
    n_questions=('question_idx', 'nunique'),
    n_models=('model', 'nunique')
).reset_index()
benchmark_flip.columns = ['Benchmark', 'Flip Rate', 'N Comparisons', 'N Questions', 'N Models']
benchmark_flip['Flip Rate'] = benchmark_flip['Flip Rate'] * 100
benchmark_flip = benchmark_flip.sort_values('Flip Rate', ascending=False)
print(benchmark_flip.to_string(index=False))

# Flip rate by model
print("\n--- Flip Rate by Model ---")
model_flip = flip_df.groupby('model')['flipped'].agg(['mean', 'count']).reset_index()
model_flip.columns = ['Model', 'Flip Rate', 'N Questions']
model_flip['Flip Rate'] = model_flip['Flip Rate'] * 100
model_flip = model_flip.sort_values('Flip Rate', ascending=False)
print(model_flip.to_string(index=False))

# Flip rate by benchmark x model
print("\n--- Flip Rate by Benchmark x Model ---")
cross_flip = flip_df.groupby(['benchmark', 'model'])['flipped'].agg(['mean', 'count']).reset_index()
cross_flip.columns = ['Benchmark', 'Model', 'Flip Rate', 'N']
cross_flip['Flip Rate'] = cross_flip['Flip Rate'] * 100
cross_pivot = cross_flip.pivot(index='Model', columns='Benchmark', values='Flip Rate')
print(cross_pivot.round(1).to_string())

# Create summary table for paper
print("\n" + "="*60)
print("SUMMARY TABLE FOR PAPER")
print("="*60)

total_questions = flip_df.groupby('benchmark')['question_idx'].nunique().sum()
total_comparisons = len(flip_df)

print("\nTable: Majority Answer Flip Rate Between T=0.7 and T=1.0")
print("-" * 55)
print(f"{'Benchmark':<20} {'Flip Rate':>10} {'Questions':>10} {'Models':>8}")
print("-" * 55)
for _, row in benchmark_flip.sort_values('Benchmark').iterrows():
    print(f"{row['Benchmark']:<20} {row['Flip Rate']:>9.1f}% {int(row['N Questions']):>10} {int(row['N Models']):>8}")
print("-" * 55)
print(f"{'Overall':<20} {overall_flip_rate:>9.1f}% {total_questions:>10} {flip_df['model'].nunique():>8}")
print("-" * 55)
print(f"\nNote: Flip rate computed over {total_comparisons} (benchmark, model, question) comparisons.")

# Check for surprising findings
print("\n" + "="*60)
print("SURPRISING FINDINGS CHECK")
print("="*60)

# High instability threshold
high_threshold = overall_flip_rate * 1.5
low_threshold = overall_flip_rate * 0.5

print(f"\nLooking for outliers (>{high_threshold:.1f}% or <{low_threshold:.1f}%):")

# Check benchmarks
for _, row in benchmark_flip.iterrows():
    if row['Flip Rate'] > high_threshold:
        print(f"  HIGH: {row['Benchmark']} has {row['Flip Rate']:.1f}% flip rate (much higher than average)")
    elif row['Flip Rate'] < low_threshold:
        print(f"  LOW: {row['Benchmark']} has {row['Flip Rate']:.1f}% flip rate (much lower than average)")

# Check models
for _, row in model_flip.iterrows():
    if row['Flip Rate'] > high_threshold:
        print(f"  HIGH: {row['Model']} has {row['Flip Rate']:.1f}% flip rate (much higher than average)")
    elif row['Flip Rate'] < low_threshold:
        print(f"  LOW: {row['Model']} has {row['Flip Rate']:.1f}% flip rate (much lower than average)")

# Save results to CSV
flip_df.to_csv('results/temperature_flip_analysis.csv', index=False)
benchmark_flip.to_csv('results/temperature_flip_by_benchmark.csv', index=False)
model_flip.to_csv('results/temperature_flip_by_model.csv', index=False)
cross_flip.to_csv('results/temperature_flip_by_benchmark_model.csv', index=False)

print("\nResults saved to results/temperature_flip_*.csv")
