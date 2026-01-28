import os
import re
import pandas as pd
import numpy as np
import wandb
from src.analyze import download_wandb_project_runs_configs, download_wandb_sweep_runs_responses, setup_notebook_dir, fix_com2sense_labels
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

REFRESH = False
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95

# Configuration and data loading
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir, results_dir = setup_notebook_dir(script_dir, refresh=False)
sweep_ids = ["4h2zjyof", "v2f7alml", "5oeyno7t", "dm625v8y", "0cn1rqtt", "6imeo6ue"]

# List of usernames to check
wandb_usernames = ['jkazdan', 'joshteam']

# Download and combine data from multiple users
all_runs_configs = []
all_responses = []

for username in wandb_usernames:
    print(f"\n{'='*60}")
    print(f"Loading data for username: {username}")
    print(f"{'='*60}")

    try:
        runs_configs = download_wandb_project_runs_configs(
            wandb_project_path="spa-experiments",
            data_dir=data_dir,
            sweep_ids=sweep_ids,
            refresh=REFRESH,
            wandb_username=username
        )
        all_runs_configs.append(runs_configs)
        print(f"✓ Loaded {len(runs_configs)} run configs from {username}")
    except Exception as e:
        print(f"⚠ Failed to load run configs from {username}: {e}")

    try:
        responses = download_wandb_sweep_runs_responses(
            wandb_project_path="spa-experiments",
            data_dir=data_dir,
            sweep_ids=sweep_ids,
            refresh=REFRESH,
            wandb_username=username
        )
        all_responses.append(responses)
        print(f"✓ Loaded {len(responses)} responses from {username}")
    except Exception as e:
        print(f"⚠ Failed to load responses from {username}: {e}")

# Combine all data
if all_runs_configs:
    runs_configs_df = pd.concat(all_runs_configs, ignore_index=True)
else:
    runs_configs_df = pd.DataFrame()
    print("⚠ No run configs loaded from any username")

if all_responses:
    responses_df = pd.concat(all_responses, ignore_index=True)
else:
    responses_df = pd.DataFrame()
    print("⚠ No responses loaded from any username")

print(f"\n{'='*60}")
print(f"COMBINED RESULTS")
print(f"{'='*60}")
print(f"Total run configs: {len(runs_configs_df)}")
print(f"Total responses: {len(responses_df)}")
if len(runs_configs_df) > 0:
    print(f"Unique runs: {runs_configs_df['run_id'].nunique()}")
if len(responses_df) > 0:
    print(f"Unique questions: {responses_df['question'].nunique() if 'question' in responses_df.columns else 'N/A'}")
    print(f"Unique models: {responses_df['config_model_id'].nunique()}")
    print(f"Response types: {responses_df['response_type'].unique()}")

# Fix com2sense labels before filtering
print(f"\n{'='*60}")
print(f"FIXING COM2SENSE LABELS")
print(f"{'='*60}")
responses_df = fix_com2sense_labels(responses_df, data_dir)

# Filter unclear answers
responses_df = responses_df[responses_df['extracted_answer'] != 'unclear']


def normalize_answer(value):
    """Normalize answer to lowercase string for consistent comparison"""
    if pd.isna(value):
        return None
    return str(value).lower().strip()


def bootstrap_accuracy(correct_array, n_bootstrap=N_BOOTSTRAP, confidence_level=CONFIDENCE_LEVEL):
    """Fast vectorized bootstrap confidence intervals for accuracy"""
    n = len(correct_array)
    if n == 0:
        return np.nan, np.nan, np.nan

    bootstrap_indices = np.random.randint(0, n, size=(n_bootstrap, n))
    bootstrap_samples = correct_array[bootstrap_indices]
    bootstrap_means = bootstrap_samples.mean(axis=1)

    mean_acc = correct_array.mean()

    alpha = 1 - confidence_level
    lower_ci = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_ci = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return mean_acc, lower_ci, upper_ci


# Process direct answers - ENSEMBLE across all models
print("\n" + "=" * 60)
print("ENSEMBLE DIRECT MAJORITY VOTE")
print("=" * 60)

def compute_direct_majority_ensemble(df):
    """Compute direct majority vote across all models"""
    direct_df = df[df['response_type'] == 'direct_answer'].copy()
    
    if len(direct_df) == 0:
        print("No direct_answer data found")
        return None
    
    # Convert answers
    mask_com2sense = direct_df['config_dataset_name'] == 'tasksource/com2sense'
    direct_df.loc[mask_com2sense, 'extracted_answer'] = direct_df.loc[mask_com2sense, 'extracted_answer'].str.lower().map({'true': True, 'false': False})
    
    mask_boolq = direct_df['config_dataset_name'] == 'google/boolq'
    direct_df.loc[mask_boolq, 'extracted_answer'] = direct_df.loc[mask_boolq, 'extracted_answer'].str.upper().map({'TRUE': True, 'FALSE': False})
    direct_df.loc[mask_boolq, 'true_answer'] = direct_df.loc[mask_boolq, 'true_answer'].apply(
        lambda x: True if (x == True or str(x).lower() == 'true') else False
    )
    
    mask_future = direct_df['config_dataset_name'] == 'kyssen/predict-the-futurebench-cutoff-June25'
    direct_df.loc[mask_future, 'extracted_answer'] = direct_df.loc[mask_future, 'extracted_answer'].str.lower()
    
    # Calculate correctness - normalize to lowercase strings for comparison
    direct_df['correct'] = (
        direct_df['extracted_answer'].apply(lambda x: str(x).lower().strip() if pd.notna(x) else None) ==
        direct_df['true_answer'].apply(lambda x: str(x).lower().strip() if pd.notna(x) else None)
    )
    
    results = []
    for dataset, group in direct_df.groupby('config_dataset_name'):
        questions = group['question'].unique()
        question_correct = {}
        
        for q in questions:
            q_data = group[group['question'] == q]
            # Simple majority vote
            vote_counts = q_data['extracted_answer'].value_counts()
            if len(vote_counts) > 0:
                majority_answer = vote_counts.idxmax()
                true_answer = q_data['true_answer'].iloc[0]
                question_correct[q] = (normalize_answer(majority_answer) == normalize_answer(true_answer))
        
        correct_array = np.array([question_correct[q] for q in questions if q in question_correct])
        
        if len(correct_array) > 0:
            mean_acc, lower_ci, upper_ci = bootstrap_accuracy(correct_array)
            results.append({
                'dataset': dataset,
                'accuracy': mean_acc * 100,
                'lower_ci': lower_ci * 100,
                'upper_ci': upper_ci * 100,
                'num_questions': len(correct_array)
            })
    
    return pd.DataFrame(results)

ensemble_direct = compute_direct_majority_ensemble(responses_df)
if ensemble_direct is not None:
    print(ensemble_direct)


# Process confidence-weighted answers - ENSEMBLE
print("\n" + "=" * 60)
print("ENSEMBLE CONFIDENCE-WEIGHTED MAJORITY VOTE")
print("=" * 60)

def compute_confidence_weighted_ensemble(df):
    """Compute confidence-weighted majority vote across all models"""
    conf_df = df[df['response_type'] == 'confidence'].copy()
    
    if len(conf_df) == 0:
        print("No confidence data found")
        return None
    
    # Extract confidence
    def extract(response):
        if pd.isna(response) or not isinstance(response, str):
            return None, None
        ans = re.search(r'Answer:\s*(\w+)', response, re.IGNORECASE)
        conf = re.search(r'Confidence:\s*([\d.]+)', response, re.IGNORECASE)
        if ans and conf:
            return (ans.group(1), float(conf.group(1)))
        return None, None
    
    extracted = conf_df['model_response'].apply(extract)
    conf_df['extracted_answer'] = extracted.apply(lambda x: x[0])
    conf_df['confidence'] = extracted.apply(lambda x: x[1])
    conf_df = conf_df.dropna(subset=['extracted_answer', 'confidence'])
    
    # Convert answers
    mask_com2sense = conf_df['config_dataset_name'] == 'tasksource/com2sense'
    conf_df.loc[mask_com2sense, 'extracted_answer'] = conf_df.loc[mask_com2sense, 'extracted_answer'].str.lower().map({'true': True, 'false': False})
    
    mask_boolq = conf_df['config_dataset_name'] == 'google/boolq'
    conf_df.loc[mask_boolq, 'extracted_answer'] = conf_df.loc[mask_boolq, 'extracted_answer'].str.upper().map({'TRUE': True, 'FALSE': False})
    
    mask_hle = conf_df['config_dataset_name'] == 'cais/hle'
    conf_df.loc[mask_hle, 'extracted_answer'] = conf_df.loc[mask_hle, 'extracted_answer'].str.lower()
    
    mask_future = conf_df['config_dataset_name'] == 'kyssen/predict-the-futurebench-cutoff-June25'
    conf_df.loc[mask_future, 'extracted_answer'] = conf_df.loc[mask_future, 'extracted_answer'].str.lower()
    
    results = []
    for dataset, group in conf_df.groupby('config_dataset_name'):
        questions = group['question'].unique()
        question_correct = {}

        for q in questions:
            q_data = group[group['question'] == q]
            # Confidence-weighted vote
            weighted_votes = q_data.groupby('extracted_answer')['confidence'].sum()
            if len(weighted_votes) > 0:
                majority_answer = weighted_votes.idxmax()
                true_answer = q_data['true_answer'].iloc[0]
                question_correct[q] = (normalize_answer(majority_answer) == normalize_answer(true_answer))
        
        correct_array = np.array([question_correct[q] for q in questions if q in question_correct])
        
        if len(correct_array) > 0:
            mean_acc, lower_ci, upper_ci = bootstrap_accuracy(correct_array)
            results.append({
                'dataset': dataset,
                'accuracy': mean_acc * 100,
                'lower_ci': lower_ci * 100,
                'upper_ci': upper_ci * 100,
                'num_questions': len(correct_array)
            })
    
    return pd.DataFrame(results)

ensemble_conf = compute_confidence_weighted_ensemble(responses_df)
if ensemble_conf is not None:
    print(ensemble_conf)


# Process prediction answers - ENSEMBLE
print("\n" + "=" * 60)
print("ENSEMBLE PREDICTION-WEIGHTED MAJORITY VOTE")
print("=" * 60)

def compute_prediction_weighted_ensemble(df):
    """Compute prediction-weighted majority vote across all models"""
    pred_df = df[df['response_type'] == 'prediction'].copy()
    
    if len(pred_df) == 0:
        print("No prediction data found")
        return None
    
    # Extract predictions
    def extract_pred(response, dataset):
        if pd.isna(response) or not isinstance(response, str):
            return None, None
        
        if dataset == 'cais/hle' or dataset == 'kyssen/predict-the-futurebench-cutoff-June25':
            yes_match = re.search(r'YES:\s*(\d+)', response, re.IGNORECASE)
            no_match = re.search(r'NO:\s*(\d+)', response, re.IGNORECASE)
            if yes_match and no_match:
                yes_conf = int(yes_match.group(1))
                no_conf = int(no_match.group(1))
                answer = 'yes' if yes_conf > no_conf else 'no'
                confidence = max(yes_conf, no_conf)
                return (answer, confidence)
        else:
            true_match = re.search(r'TRUE:\s*(\d+)', response, re.IGNORECASE)
            false_match = re.search(r'FALSE:\s*(\d+)', response, re.IGNORECASE)
            if true_match and false_match:
                true_conf = int(true_match.group(1))
                false_conf = int(false_match.group(1))
                answer = True if true_conf > false_conf else False
                confidence = max(true_conf, false_conf)
                return (answer, confidence)
        
        return None, None
    
    extracted = pred_df.apply(lambda row: extract_pred(row['model_response'], row['config_dataset_name']), axis=1)
    pred_df['extracted_answer'] = extracted.apply(lambda x: x[0])
    pred_df['confidence'] = extracted.apply(lambda x: x[1])
    pred_df = pred_df.dropna(subset=['extracted_answer', 'confidence'])
    
    # Convert true_answer for boolq
    mask_boolq = pred_df['config_dataset_name'] == 'google/boolq'
    pred_df.loc[mask_boolq, 'true_answer'] = pred_df.loc[mask_boolq, 'true_answer'].apply(
        lambda x: True if (x == True or str(x).lower() == 'true') else False
    )
    
    results = []
    for dataset, group in pred_df.groupby('config_dataset_name'):
        questions = group['question'].unique()
        question_correct = {}

        for q in questions:
            q_data = group[group['question'] == q]
            # Confidence-weighted vote
            weighted_votes = q_data.groupby('extracted_answer')['confidence'].sum()
            if len(weighted_votes) > 0:
                majority_answer = weighted_votes.idxmax()
                true_answer = q_data['true_answer'].iloc[0]
                question_correct[q] = (normalize_answer(majority_answer) == normalize_answer(true_answer))
        
        correct_array = np.array([question_correct[q] for q in questions if q in question_correct])
        
        if len(correct_array) > 0:
            mean_acc, lower_ci, upper_ci = bootstrap_accuracy(correct_array)
            results.append({
                'dataset': dataset,
                'accuracy': mean_acc * 100,
                'lower_ci': lower_ci * 100,
                'upper_ci': upper_ci * 100,
                'num_questions': len(correct_array)
            })
    
    return pd.DataFrame(results)

ensemble_pred = compute_prediction_weighted_ensemble(responses_df)
if ensemble_pred is not None:
    print(ensemble_pred)


# Compute highest confidence selection - ENSEMBLE across all models
print("\n" + "=" * 60)
print("ENSEMBLE HIGHEST CONFIDENCE SELECTION")
print("=" * 60)

def compute_ensemble_highest_confidence(df):
    """For each question, select the answer with highest confidence across ALL models"""
    conf_df = df[df['response_type'] == 'confidence'].copy()
    if len(conf_df) == 0:
        print("No confidence data found")
        return None
    
    # Extract confidence
    def extract(response):
        if pd.isna(response) or not isinstance(response, str):
            return None, None
        ans = re.search(r'Answer:\s*(\w+)', response, re.IGNORECASE)
        conf = re.search(r'Confidence:\s*([\d.]+)', response, re.IGNORECASE)
        if ans and conf:
            return (ans.group(1), float(conf.group(1)))
        return None, None
    
    extracted = conf_df['model_response'].apply(extract)
    conf_df['extracted_answer'] = extracted.apply(lambda x: x[0])
    conf_df['confidence'] = extracted.apply(lambda x: x[1])
    conf_df = conf_df.dropna(subset=['extracted_answer', 'confidence'])
    
    # Convert answers
    mask_com2sense = conf_df['config_dataset_name'] == 'tasksource/com2sense'
    conf_df.loc[mask_com2sense, 'extracted_answer'] = conf_df.loc[mask_com2sense, 'extracted_answer'].str.lower().map({'true': True, 'false': False})
    
    mask_boolq = conf_df['config_dataset_name'] == 'google/boolq'
    conf_df.loc[mask_boolq, 'extracted_answer'] = conf_df.loc[mask_boolq, 'extracted_answer'].str.upper().map({'TRUE': True, 'FALSE': False})
    
    mask_hle = conf_df['config_dataset_name'] == 'cais/hle'
    conf_df.loc[mask_hle, 'extracted_answer'] = conf_df.loc[mask_hle, 'extracted_answer'].str.lower()
    
    mask_future = conf_df['config_dataset_name'] == 'kyssen/predict-the-futurebench-cutoff-June25'
    conf_df.loc[mask_future, 'extracted_answer'] = conf_df.loc[mask_future, 'extracted_answer'].str.lower()
    
    results = []
    for dataset, group in conf_df.groupby('config_dataset_name'):
        questions = group['question'].unique()
        question_correct = {}
        
        for q in questions:
            q_data = group[group['question'] == q]
            max_idx = q_data['confidence'].idxmax()
            selected_answer = q_data.loc[max_idx, 'extracted_answer']
            true_answer = q_data['true_answer'].iloc[0]
            question_correct[q] = (normalize_answer(selected_answer) == normalize_answer(true_answer))
        
        correct_array = np.array([question_correct[q] for q in questions])
        
        if len(correct_array) > 0:
            mean_acc, lower_ci, upper_ci = bootstrap_accuracy(correct_array)
            results.append({
                'dataset': dataset,
                'accuracy': mean_acc * 100,
                'lower_ci': lower_ci * 100,
                'upper_ci': upper_ci * 100,
                'num_questions': len(correct_array)
            })
    
    return pd.DataFrame(results)

ensemble_highest_conf = compute_ensemble_highest_confidence(responses_df)
if ensemble_highest_conf is not None:
    print(ensemble_highest_conf)


# Compute Surprisingly Popular Answer - ENSEMBLE across all models
print("\n" + "=" * 60)
print("ENSEMBLE SURPRISINGLY POPULAR ANSWER")
print("=" * 60)

def compute_ensemble_surprisingly_popular(df):
    """Compute surprisingly popular answer across ALL models"""
    
    direct_df = df[df['response_type'] == 'direct_answer'].copy()
    if len(direct_df) == 0:
        print("No direct_answer data found")
        return None
    
    pred_df = df[df['response_type'] == 'prediction'].copy()
    if len(pred_df) == 0:
        print("No prediction data found")
        return None
    
    # Extract predictions
    def extract_both_predictions(response, dataset):
        if pd.isna(response) or not isinstance(response, str):
            return None, None
        
        if dataset == 'cais/hle' or dataset == 'kyssen/predict-the-futurebench-cutoff-June25':
            yes_match = re.search(r'YES:\s*(\d+)', response, re.IGNORECASE)
            no_match = re.search(r'NO:\s*(\d+)', response, re.IGNORECASE)
            if yes_match and no_match:
                return (int(yes_match.group(1)), int(no_match.group(1)))
        else:
            true_match = re.search(r'TRUE:\s*(\d+)', response, re.IGNORECASE)
            false_match = re.search(r'FALSE:\s*(\d+)', response, re.IGNORECASE)
            if true_match and false_match:
                return (int(true_match.group(1)), int(false_match.group(1)))
        
        return None, None
    
    extracted_pred = pred_df.apply(
        lambda row: extract_both_predictions(row['model_response'], row['config_dataset_name']), 
        axis=1
    )
    pred_df['pred_true_conf'] = extracted_pred.apply(lambda x: x[0])
    pred_df['pred_false_conf'] = extracted_pred.apply(lambda x: x[1])
    pred_df = pred_df.dropna(subset=['pred_true_conf', 'pred_false_conf'])
    
    # Convert direct answers to binary
    direct_df['answer_binary'] = 0
    
    for dataset in ['cais/hle', 'kyssen/predict-the-futurebench-cutoff-June25']:
        dataset_mask = direct_df['config_dataset_name'] == dataset
        direct_df.loc[dataset_mask & direct_df['extracted_answer'].isin(['yes']), 'answer_binary'] = 1
    
    for dataset in ['google/boolq', 'tasksource/com2sense']:
        dataset_mask = direct_df['config_dataset_name'] == dataset
        direct_df.loc[dataset_mask & direct_df['extracted_answer'].isin(['true', True]), 'answer_binary'] = 1
    
    results = []
    for dataset in direct_df['config_dataset_name'].unique():
        d_group = direct_df[direct_df['config_dataset_name'] == dataset]
        p_group = pred_df[pred_df['config_dataset_name'] == dataset]
        
        if len(p_group) == 0:
            continue
        
        common_questions = list(set(d_group['question'].unique()) & set(p_group['question'].unique()))
        
        if len(common_questions) == 0:
            continue
        
        question_correct = {}
        for q in common_questions:
            q_direct = d_group[d_group['question'] == q]
            q_pred = p_group[p_group['question'] == q]
            
            avg_direct = q_direct['answer_binary'].mean()
            avg_pred_true = q_pred['pred_true_conf'].mean()
            avg_pred_false = q_pred['pred_false_conf'].mean()
            
            gap_true = avg_direct - (avg_pred_true / 100)
            gap_false = (1 - avg_direct) - (avg_pred_false / 100)
            
            sp_answer = True if gap_true > gap_false else False

            if dataset in ['cais/hle', 'kyssen/predict-the-futurebench-cutoff-June25']:
                sp_answer = 'yes' if sp_answer else 'no'

            true_answer = q_direct['true_answer'].iloc[0]

            question_correct[q] = (normalize_answer(sp_answer) == normalize_answer(true_answer))
        
        correct_array = np.array([question_correct[q] for q in common_questions])
        
        if len(correct_array) > 0:
            mean_acc, lower_ci, upper_ci = bootstrap_accuracy(correct_array)
            results.append({
                'dataset': dataset,
                'accuracy': mean_acc * 100,
                'lower_ci': lower_ci * 100,
                'upper_ci': upper_ci * 100,
                'num_questions': len(correct_array)
            })
    
    return pd.DataFrame(results)

ensemble_sp = compute_ensemble_surprisingly_popular(responses_df)
if ensemble_sp is not None:
    print(ensemble_sp)


# Create summary comparison
print("\n" + "=" * 60)
print("ENSEMBLE SUMMARY")
print("=" * 60)

# Combine all ensemble results
all_ensemble_results = []

if ensemble_direct is not None and len(ensemble_direct) > 0:
    temp = ensemble_direct.copy()
    temp['method'] = 'Direct Majority'
    all_ensemble_results.append(temp)

if ensemble_conf is not None and len(ensemble_conf) > 0:
    temp = ensemble_conf.copy()
    temp['method'] = 'Conf Weighted'
    all_ensemble_results.append(temp)

if ensemble_pred is not None and len(ensemble_pred) > 0:
    temp = ensemble_pred.copy()
    temp['method'] = 'Pred Weighted'
    all_ensemble_results.append(temp)

if ensemble_highest_conf is not None and len(ensemble_highest_conf) > 0:
    temp = ensemble_highest_conf.copy()
    temp['method'] = 'Highest Conf'
    all_ensemble_results.append(temp)

if ensemble_sp is not None and len(ensemble_sp) > 0:
    temp = ensemble_sp.copy()
    temp['method'] = 'Surp. Popular'
    all_ensemble_results.append(temp)

if all_ensemble_results:
    ensemble_summary = pd.concat(all_ensemble_results, ignore_index=True)
    
    print("\nEnsemble Summary DataFrame:")
    print(ensemble_summary)
    
    # Create visualization
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    
    palette = {
        'Direct Majority': '#3498db',
        'Highest Conf': '#9b59b6',
        'Conf Weighted': '#e74c3c', 
        'Pred Weighted': '#2ecc71',
        'Surp. Popular': '#f39c12'
    }
    
    # Create plot with larger figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Get unique datasets and methods
    datasets = sorted(list(set([str(d) for d in ensemble_summary['dataset'].unique()])))
    methods = ['Direct Majority', 'Highest Conf', 'Conf Weighted', 'Pred Weighted', 'Surp. Popular']
    methods = [m for m in methods if m in ensemble_summary['method'].values]
    
    print(f"\nDatasets found: {datasets}")
    print(f"Methods found: {methods}")
    
    x = np.arange(len(datasets))
    width = 0.15
    
    for i, method in enumerate(methods):
        method_data = ensemble_summary[ensemble_summary['method'] == method].copy()
        
        # Ensure dataset column is string
        method_data['dataset'] = method_data['dataset'].astype(str)
        
        # Reindex to match all datasets
        method_data = method_data.set_index('dataset').reindex(datasets).reset_index()
        
        positions = x + (i - len(methods)/2 + 0.5) * width
        
        # Plot bars
        bars = ax.bar(positions, method_data['accuracy'], width, 
                     label=method, color=palette.get(method, '#95a5a6'),
                     alpha=0.85, edgecolor='white', linewidth=2)
        
        # Add error bars
        errors_lower = method_data['accuracy'] - method_data['lower_ci']
        errors_upper = method_data['upper_ci'] - method_data['accuracy']
        
        ax.errorbar(positions, method_data['accuracy'],
                   yerr=[errors_lower, errors_upper],
                   fmt='none', ecolor='black', elinewidth=2,
                   capsize=4, capthick=2, alpha=0.7)
        
        # Add value labels on top of bars
        for pos, val in zip(positions, method_data['accuracy']):
            if not pd.isna(val):
                ax.text(pos, val + 2, f'{val:.1f}', 
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    # Customize plot
    dataset_labels = [d.split('/')[-1].replace('-', '\n') for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Ensemble Methods Across All Models\n(with {int(CONFIDENCE_LEVEL*100)}% Bootstrap Confidence Intervals)',
        fontsize=18, fontweight='bold', pad=20
    )
    ax.set_ylim(0, 110)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, ncol=1)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{results_dir}/ensemble_comparison_all_models.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {filename}")
    plt.close()
    
    # Save results table
    summary_filename = f"{results_dir}/ensemble_results_all_models.csv"
    ensemble_summary.to_csv(summary_filename, index=False)
    print(f"Saved: {summary_filename}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS TABLE")
    print("=" * 60)
    pivot_table = ensemble_summary.pivot_table(
        index='dataset',
        columns='method',
        values='accuracy',
        aggfunc='first'
    )
    print(pivot_table.to_string())
else:
    print("\n⚠ No ensemble results to visualize!")

print("\n" + "=" * 60)
print("ENSEMBLE ANALYSIS COMPLETE")
print("=" * 60)