#!/usr/bin/env python3
"""
Compute paper statistics from local CSV files.
This script loads the pre-downloaded data and computes all correlation statistics
needed for the manuscript.
"""

import os
import re
import pandas as pd
import numpy as np
from scipy import stats

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
results_dir = os.path.join(script_dir, 'results')

print("="*60)
print("LOADING LOCAL DATA")
print("="*60)

# Load all response CSV files from data directory
all_responses = []
all_configs = []

for filename in os.listdir(data_dir):
    if filename.endswith('_responses.csv'):
        filepath = os.path.join(data_dir, filename)
        print(f"Loading: {filename}")
        df = pd.read_csv(filepath)
        all_responses.append(df)
        print(f"  -> {len(df)} responses")
    elif filename.endswith('_runs_configs.csv'):
        filepath = os.path.join(data_dir, filename)
        print(f"Loading: {filename}")
        df = pd.read_csv(filepath)
        all_configs.append(df)
        print(f"  -> {len(df)} configs")

# Combine data
responses_df = pd.concat(all_responses, ignore_index=True) if all_responses else pd.DataFrame()
print(f"\nTotal responses loaded: {len(responses_df)}")

# Normalize data
print("\n" + "="*60)
print("NORMALIZING DATA TYPES")
print("="*60)

responses_df['extracted_answer'] = responses_df['extracted_answer'].apply(
    lambda x: str(x).lower() if pd.notna(x) else None
)
responses_df['true_answer'] = responses_df['true_answer'].apply(
    lambda x: str(x).lower() if pd.notna(x) else None
)

# Filter unclear answers
responses_df = responses_df[responses_df['extracted_answer'] != 'unclear'].copy()
responses_df = responses_df.dropna(subset=['extracted_answer', 'true_answer'])

print(f"Total valid responses after filtering: {len(responses_df)}")

# Check what response types exist
print("\n" + "="*60)
print("DATA AVAILABILITY BY DATASET")
print("="*60)
response_type_by_dataset = responses_df.groupby(['config_dataset_name', 'response_type']).size().unstack(fill_value=0)
print(response_type_by_dataset)

# Helper functions
def extract_both_predictions(response, dataset):
    """Extract prediction fractions for both options"""
    if pd.isna(response) or not isinstance(response, str):
        return None, None

    if dataset in ['cais/hle', 'kyssen/predict-the-futurebench-cutoff-June25']:
        yes_match = re.search(r'YES:?\s*(\d+)', response, re.I)
        no_match = re.search(r'NO:?\s*(\d+)', response, re.I)
        if yes_match and no_match:
            return (int(yes_match.group(1)), int(no_match.group(1)))
    else:
        true_match = re.search(r'TRUE:?\s*(\d+)', response, re.I)
        false_match = re.search(r'FALSE:?\s*(\d+)', response, re.I)
        if true_match and false_match:
            return (int(true_match.group(1)), int(false_match.group(1)))

    return None, None

def extract_confidence(response):
    """Extract answer and confidence from response"""
    if pd.isna(response) or not isinstance(response, str):
        return None, None

    ans = re.search(r'Answer:?\s*(yes|no|true|false)', response, re.I)
    conf = re.search(r'Confidence:?\s*(\d+)', response, re.I)

    if ans and conf:
        return (ans.group(1).lower(), float(conf.group(1)))
    return None, None

# ============================================================================
# PROCESS CONFIDENCE DATA
# ============================================================================
print("\n" + "="*60)
print("PROCESSING CONFIDENCE DATA")
print("="*60)

conf_df = responses_df[responses_df['response_type'] == 'confidence'].copy()

if len(conf_df) > 0:
    # Extract confidence
    extracted = conf_df['model_response'].apply(extract_confidence)
    conf_df['extracted_answer'] = extracted.apply(lambda x: x[0])
    conf_df['confidence'] = extracted.apply(lambda x: x[1])
    conf_df = conf_df.dropna(subset=['extracted_answer', 'confidence'])

    # Normalize answers
    conf_df['extracted_answer'] = conf_df['extracted_answer'].str.lower()
    conf_df['correct'] = (conf_df['extracted_answer'] == conf_df['true_answer'])

    print(f"Total confidence responses: {len(conf_df)}")
else:
    print("No confidence data available")

# ============================================================================
# PROCESS DIRECT ANSWER AND PREDICTION DATA
# ============================================================================
print("\n" + "="*60)
print("PROCESSING DIRECT ANSWER AND PREDICTION DATA")
print("="*60)

direct_df = responses_df[responses_df['response_type'] == 'direct_answer'].copy()
pred_df = responses_df[responses_df['response_type'] == 'prediction'].copy()

print(f"Direct answers: {len(direct_df)}")
print(f"Predictions: {len(pred_df)}")

# Process predictions
if len(pred_df) > 0:
    extracted_pred = pred_df.apply(
        lambda row: extract_both_predictions(row['model_response'], row['config_dataset_name']),
        axis=1
    )
    pred_df['pred_option1_conf'] = extracted_pred.apply(lambda x: x[0])
    pred_df['pred_option2_conf'] = extracted_pred.apply(lambda x: x[1])
    pred_df = pred_df.dropna(subset=['pred_option1_conf', 'pred_option2_conf'])
    print(f"Valid predictions: {len(pred_df)}")

# Process direct answers
if len(direct_df) > 0:
    direct_df['answer_binary'] = 0
    for dataset in ['cais/hle', 'kyssen/predict-the-futurebench-cutoff-June25']:
        dataset_mask = direct_df['config_dataset_name'] == dataset
        direct_df.loc[dataset_mask & (direct_df['extracted_answer'] == 'yes'), 'answer_binary'] = 1
    for dataset in ['google/boolq', 'tasksource/com2sense']:
        dataset_mask = direct_df['config_dataset_name'] == dataset
        direct_df.loc[dataset_mask & (direct_df['extracted_answer'] == 'true'), 'answer_binary'] = 1

# ============================================================================
# COMPUTE AGREEMENT DATA
# ============================================================================
print("\n" + "="*60)
print("COMPUTING AGREEMENT DATA")
print("="*60)

agreement_data = []

if len(conf_df) > 0:
    for (dataset, model, temp, question), group in conf_df.groupby(['config_dataset_name', 'config_model_id', 'config_temperature', 'question']):
        answer_counts = group['extracted_answer'].value_counts()
        if len(answer_counts) == 0:
            continue

        most_popular = answer_counts.idxmax()
        popular_responses = group[group['extracted_answer'] == most_popular]
        if len(popular_responses) == 0:
            continue

        mean_confidence = popular_responses['confidence'].mean()
        agreement_fraction = len(popular_responses) / len(group)

        agreement_data.append({
            'dataset': dataset,
            'model': model,
            'temperature': temp,
            'question': question,
            'mean_confidence': mean_confidence / 100,
            'agreement_fraction': agreement_fraction
        })

agreement_df = pd.DataFrame(agreement_data)
print(f"Total agreement questions: {len(agreement_df)}")

# ============================================================================
# COMPUTE PREDICTED POPULARITY DATA
# ============================================================================
print("\n" + "="*60)
print("COMPUTING PREDICTED POPULARITY DATA")
print("="*60)

pred_agreement_data = []

if len(direct_df) > 0 and len(pred_df) > 0:
    for (dataset, model, temp), d_group in direct_df.groupby(['config_dataset_name', 'config_model_id', 'config_temperature']):
        p_group = pred_df[(pred_df['config_dataset_name'] == dataset) &
                          (pred_df['config_model_id'] == model) &
                          (pred_df['config_temperature'] == temp)]

        if len(p_group) == 0:
            continue

        common_questions = list(set(d_group['question'].unique()) & set(p_group['question'].unique()))

        for q in common_questions:
            q_direct = d_group[d_group['question'] == q]
            q_pred = p_group[p_group['question'] == q]

            true_answer = q_direct['true_answer'].iloc[0]
            agreement_with_correct = (q_direct['extracted_answer'] == true_answer).mean()
            avg_pred_option1 = q_pred['pred_option1_conf'].mean() / 100.0

            if dataset in ['cais/hle', 'kyssen/predict-the-futurebench-cutoff-June25']:
                predicted_popularity = avg_pred_option1 if true_answer == 'yes' else (1 - avg_pred_option1)
            else:
                predicted_popularity = avg_pred_option1 if true_answer == 'true' else (1 - avg_pred_option1)

            pred_agreement_data.append({
                'dataset': dataset,
                'model': model,
                'temperature': temp,
                'question': q,
                'predicted_popularity': predicted_popularity,
                'agreement_with_correct': agreement_with_correct
            })

pred_agreement_df = pd.DataFrame(pred_agreement_data)
print(f"Total predicted popularity questions: {len(pred_agreement_df)}")

# ============================================================================
# COMPUTE PAPER STATISTICS
# ============================================================================
print("\n" + "="*60)
print("COMPUTING CORRELATION STATISTICS FOR PAPER")
print("="*60)

paper_stats = {}

# 1. Confidence vs Accuracy correlation
if len(conf_df) > 0:
    conf_df['correct_binary'] = conf_df['correct'].astype(float)
    r_conf_acc = conf_df['confidence'].corr(conf_df['correct_binary'])
    paper_stats['r_confidence_accuracy'] = r_conf_acc
    print(f"\n1. Correlation (confidence, accuracy): r = {r_conf_acc:.3f}")

    # Per-dataset breakdown
    print("   Per-dataset breakdown:")
    for dataset in sorted(conf_df['config_dataset_name'].unique()):
        ds_data = conf_df[conf_df['config_dataset_name'] == dataset]
        r_ds = ds_data['confidence'].corr(ds_data['correct_binary'])
        ds_short = dataset.split('/')[-1]
        paper_stats[f'r_conf_acc_{ds_short}'] = r_ds
        print(f"     {ds_short}: r = {r_ds:.3f}")

    # Accuracy in 90-100% confidence bin
    high_conf = conf_df[conf_df['confidence'] >= 90]
    if len(high_conf) > 0:
        acc_high_conf = high_conf['correct'].mean() * 100
        paper_stats['accuracy_at_90plus_confidence'] = acc_high_conf
        print(f"\n2. Accuracy at 90%+ confidence: {acc_high_conf:.1f}%")

        # Per-dataset
        print("   Per-dataset breakdown:")
        for dataset in sorted(high_conf['config_dataset_name'].unique()):
            ds_data = high_conf[high_conf['config_dataset_name'] == dataset]
            acc_ds = ds_data['correct'].mean() * 100
            n_samples = len(ds_data)
            ds_short = dataset.split('/')[-1]
            paper_stats[f'acc_90plus_{ds_short}'] = acc_ds
            print(f"     {ds_short}: {acc_ds:.1f}% (n={n_samples})")

# 2. Confidence vs Agreement correlation
if len(agreement_df) > 0:
    r_conf_agree = agreement_df['mean_confidence'].corr(agreement_df['agreement_fraction'])
    paper_stats['r_confidence_agreement'] = r_conf_agree
    print(f"\n3. Correlation (confidence, agreement): r = {r_conf_agree:.3f}")

    # Per-model breakdown
    print("   Per-model breakdown:")
    for model in sorted(agreement_df['model'].unique()):
        model_data = agreement_df[agreement_df['model'] == model]
        r_model = model_data['mean_confidence'].corr(model_data['agreement_fraction'])
        model_short = model.split('.')[-1].split(':')[0]
        paper_stats[f'r_conf_agree_{model_short}'] = r_model
        print(f"     {model_short}: r = {r_model:.3f}")

# 3. Predicted vote vs Actual vote correlation and MAE
if len(pred_agreement_df) > 0:
    r_pred_actual = pred_agreement_df['predicted_popularity'].corr(
        pred_agreement_df['agreement_with_correct'])
    mae_pred = (pred_agreement_df['predicted_popularity'] -
                pred_agreement_df['agreement_with_correct']).abs().mean()
    paper_stats['r_predicted_actual_vote'] = r_pred_actual
    paper_stats['mae_predicted_actual'] = mae_pred
    print(f"\n4. Correlation (predicted vote, actual vote): r = {r_pred_actual:.3f}")
    print(f"5. MAE (predicted vs actual): {mae_pred:.3f}")

    # Per-dataset breakdown
    print("\n   Per-dataset breakdown:")
    for dataset in sorted(pred_agreement_df['dataset'].unique()):
        ds_data = pred_agreement_df[pred_agreement_df['dataset'] == dataset]
        r_ds = ds_data['predicted_popularity'].corr(ds_data['agreement_with_correct'])
        mae_ds = (ds_data['predicted_popularity'] - ds_data['agreement_with_correct']).abs().mean()
        ds_short = dataset.split('/')[-1]
        paper_stats[f'r_pred_actual_{ds_short}'] = r_ds
        paper_stats[f'mae_pred_{ds_short}'] = mae_ds
        print(f"     {ds_short}: r = {r_ds:.3f}, MAE = {mae_ds:.3f}")

# ============================================================================
# SP MAJORITY SELECTION ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("SP MAJORITY SELECTION ANALYSIS")
print("="*60)

if len(direct_df) > 0 and len(pred_df) > 0:
    sp_majority_analysis = []

    for (dataset, model, temp), d_group in direct_df.groupby(['config_dataset_name', 'config_model_id', 'config_temperature']):
        if dataset != 'cais/hle':
            continue

        p_group = pred_df[(pred_df['config_dataset_name'] == dataset) &
                          (pred_df['config_model_id'] == model) &
                          (pred_df['config_temperature'] == temp)]

        if len(p_group) == 0:
            continue

        common_questions = list(set(d_group['question'].unique()) & set(p_group['question'].unique()))

        for q in common_questions:
            q_direct = d_group[d_group['question'] == q]
            q_pred = p_group[p_group['question'] == q]

            # Get majority answer
            answer_counts = q_direct['extracted_answer'].value_counts()
            majority_answer = answer_counts.idxmax()

            # Get SP answer
            avg_direct = q_direct['answer_binary'].mean()
            avg_pred_option1 = q_pred['pred_option1_conf'].mean() / 100.0
            avg_pred_option2 = q_pred['pred_option2_conf'].mean() / 100.0

            gap_option1 = avg_direct - avg_pred_option1
            gap_option2 = (1 - avg_direct) - avg_pred_option2

            sp_answer = 'yes' if gap_option1 > gap_option2 else 'no'

            sp_majority_analysis.append({
                'question': q,
                'model': model,
                'temperature': temp,
                'majority_answer': majority_answer,
                'sp_answer': sp_answer,
                'sp_selects_majority': sp_answer == majority_answer,
                'true_answer': q_direct['true_answer'].iloc[0],
                'sp_correct': sp_answer == q_direct['true_answer'].iloc[0],
                'majority_correct': majority_answer == q_direct['true_answer'].iloc[0]
            })

    sp_majority_df = pd.DataFrame(sp_majority_analysis)

    if len(sp_majority_df) > 0:
        sp_selects_majority_pct = sp_majority_df['sp_selects_majority'].mean() * 100
        sp_correct_when_minority = sp_majority_df[~sp_majority_df['sp_selects_majority']]['sp_correct'].mean() * 100 if len(sp_majority_df[~sp_majority_df['sp_selects_majority']]) > 0 else 0
        sp_correct_when_majority = sp_majority_df[sp_majority_df['sp_selects_majority']]['sp_correct'].mean() * 100 if len(sp_majority_df[sp_majority_df['sp_selects_majority']]) > 0 else 0

        paper_stats['sp_selects_majority_pct_hle'] = sp_selects_majority_pct
        paper_stats['sp_correct_when_minority_hle'] = sp_correct_when_minority
        paper_stats['sp_correct_when_majority_hle'] = sp_correct_when_majority

        print(f"\nHLE SP Majority Selection Analysis:")
        print(f"  SP selects majority answer: {sp_selects_majority_pct:.1f}% of questions")
        print(f"  SP selects minority answer: {100 - sp_selects_majority_pct:.1f}% of questions")
        print(f"\n  When SP selects majority: {sp_correct_when_majority:.1f}% correct (n={len(sp_majority_df[sp_majority_df['sp_selects_majority']])})")
        print(f"  When SP selects minority: {sp_correct_when_minority:.1f}% correct (n={len(sp_majority_df[~sp_majority_df['sp_selects_majority']])})")

# ============================================================================
# SAVE STATISTICS
# ============================================================================
print("\n" + "="*60)
print("SAVING STATISTICS")
print("="*60)

# Save to CSV
paper_stats_df = pd.DataFrame([paper_stats])
stats_filename = os.path.join(results_dir, 'paper_statistics.csv')
paper_stats_df.to_csv(stats_filename, index=False)
print(f"Saved paper statistics to: {stats_filename}")

# Save as formatted text
stats_txt_filename = os.path.join(results_dir, 'paper_statistics.txt')
with open(stats_txt_filename, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("PAPER STATISTICS SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write("SECTION 5.2 - CONFIDENCE CALIBRATION\n")
    f.write("-" * 40 + "\n")
    if 'r_confidence_accuracy' in paper_stats:
        f.write(f"Correlation (confidence, accuracy): r = {paper_stats['r_confidence_accuracy']:.3f}\n")
    if 'accuracy_at_90plus_confidence' in paper_stats:
        f.write(f"Accuracy at 90%+ confidence: {paper_stats['accuracy_at_90plus_confidence']:.1f}%\n")
    if 'r_confidence_agreement' in paper_stats:
        f.write(f"Correlation (confidence, agreement): r = {paper_stats['r_confidence_agreement']:.3f}\n")

    f.write("\nSECTION 5.3 - SOCIAL PREDICTION\n")
    f.write("-" * 40 + "\n")
    if 'r_predicted_actual_vote' in paper_stats:
        f.write(f"Correlation (predicted vote, actual vote): r = {paper_stats['r_predicted_actual_vote']:.3f}\n")
    if 'mae_predicted_actual' in paper_stats:
        f.write(f"MAE (predicted vs actual): {paper_stats['mae_predicted_actual']:.3f}\n")

    f.write("\nSECTION 5.1 - SP MAJORITY SELECTION (HLE)\n")
    f.write("-" * 40 + "\n")
    if 'sp_selects_majority_pct_hle' in paper_stats:
        f.write(f"SP selects majority answer: {paper_stats['sp_selects_majority_pct_hle']:.1f}%\n")
        f.write(f"SP correct when selecting minority: {paper_stats.get('sp_correct_when_minority_hle', 'N/A'):.1f}%\n")
        f.write(f"SP correct when selecting majority: {paper_stats.get('sp_correct_when_majority_hle', 'N/A'):.1f}%\n")

    f.write("\n" + "=" * 60 + "\n")
    f.write("FULL STATISTICS\n")
    f.write("=" * 60 + "\n")
    for key, value in sorted(paper_stats.items()):
        if isinstance(value, float):
            f.write(f"{key}: {value:.4f}\n")
        else:
            f.write(f"{key}: {value}\n")

print(f"Saved formatted statistics to: {stats_txt_filename}")

print("\n" + "="*60)
print("ALL STATISTICS COMPUTED")
print("="*60)
