import os
import re
import pandas as pd
import numpy as np
import wandb
from src.analyze import download_wandb_project_runs_configs, download_wandb_sweep_runs_responses, setup_notebook_dir, fix_com2sense_labels
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

REFRESH = False  # Set to True to re-download data
N_BINS = 5  # Number of bins for calibration plots (reduced from 10 to get more samples per bin)
MIN_SAMPLES_PER_BIN = 10  # Minimum samples required per bin to compute CI (reduced from 20 for smaller datasets)
CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals

# Configuration and data loading
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir, results_dir = setup_notebook_dir(script_dir, refresh=REFRESH)
sweep_ids = ["4h2zjyof", "v2f7alml", "5oeyno7t", "dm625v8y", "0cn1rqtt", "6imeo6ue"]

# List of usernames to check
wandb_usernames = ['jkazdan', 'joshteam']

print("="*60)
print("LOADING DATA")
print("="*60)

# Download and combine data from multiple users
all_runs_configs = []
all_responses = []

for username in wandb_usernames:
    print(f"\nLoading data for username: {username}")

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

print(f"\nTotal run configs: {len(runs_configs_df)}")
print(f"Total responses: {len(responses_df)}")

# Fix com2sense labels before processing
print("\n" + "="*60)
print("FIXING COM2SENSE LABELS")
print("="*60)
responses_df = fix_com2sense_labels(responses_df, data_dir)

# CRITICAL: Normalize all answers to lowercase strings for consistent comparison
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

# Check what response types exist for each dataset
print("\n" + "="*60)
print("DATA AVAILABILITY BY DATASET")
print("="*60)
response_type_by_dataset = responses_df.groupby(['config_dataset_name', 'response_type']).size().unstack(fill_value=0)
print(response_type_by_dataset)

# Set style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.0)

# Define consistent color palettes
dataset_palette = {
    'cais/hle': '#e74c3c',
    'google/boolq': '#3498db',
    'kyssen/predict-the-futurebench-cutoff-June25': '#2ecc71',
    'tasksource/com2sense': '#f39c12'
}

# Get unique models and assign colors
unique_models = sorted(responses_df['config_model_id'].unique())
model_colors = sns.color_palette("husl", n_colors=len(unique_models))
model_palette = {model: color for model, color in zip(unique_models, model_colors)}

print(f"\nDatasets: {sorted(responses_df['config_dataset_name'].unique())}")
print(f"Models: {unique_models}")

# Helper function for parametric confidence intervals using SEM
def parametric_ci(data, confidence_level=CONFIDENCE_LEVEL):
    """
    Calculate parametric confidence intervals for the mean using SEM.
    CI = mean ± z * SEM, where SEM = SD / sqrt(n)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    n = len(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)  # Standard error of the mean

    # For 95% CI, z = 1.96
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)

    lower_ci = mean - z_score * sem
    upper_ci = mean + z_score * sem

    return mean, lower_ci, upper_ci

# ============================================================================
# Helper functions for data extraction
# ============================================================================

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
# 1. SURPRISE GAP VS SPA ACCURACY (Line Graph) with Bootstrap CIs
# ============================================================================
print("\n" + "="*60)
print("GRAPH 1: SURPRISE GAP VS SPA ACCURACY")
print("="*60)

spa_data = []
direct_df = responses_df[responses_df['response_type'] == 'direct_answer'].copy()
pred_df = responses_df[responses_df['response_type'] == 'prediction'].copy()

if len(direct_df) > 0 and len(pred_df) > 0:
    # Extract predictions
    extracted_pred = pred_df.apply(
        lambda row: extract_both_predictions(row['model_response'], row['config_dataset_name']),
        axis=1
    )
    pred_df['pred_option1_conf'] = extracted_pred.apply(lambda x: x[0])
    pred_df['pred_option2_conf'] = extracted_pred.apply(lambda x: x[1])
    pred_df = pred_df.dropna(subset=['pred_option1_conf', 'pred_option2_conf'])

    # Convert direct answers to binary (1 for yes/true, 0 for no/false)
    direct_df['answer_binary'] = 0

    for dataset in ['cais/hle', 'kyssen/predict-the-futurebench-cutoff-June25']:
        dataset_mask = direct_df['config_dataset_name'] == dataset
        direct_df.loc[dataset_mask & (direct_df['extracted_answer'] == 'yes'), 'answer_binary'] = 1

    for dataset in ['google/boolq', 'tasksource/com2sense']:
        dataset_mask = direct_df['config_dataset_name'] == dataset
        direct_df.loc[dataset_mask & (direct_df['extracted_answer'] == 'true'), 'answer_binary'] = 1

    # Calculate SPA for each question
    for (dataset, model, temp), d_group in direct_df.groupby(['config_dataset_name', 'config_model_id', 'config_temperature']):
        p_group = pred_df[(pred_df['config_dataset_name'] == dataset) &
                          (pred_df['config_model_id'] == model) &
                          (pred_df['config_temperature'] == temp)]

        if len(p_group) == 0:
            continue

        # Get common questions
        common_questions = list(set(d_group['question'].unique()) & set(p_group['question'].unique()))

        for q in common_questions:
            q_direct = d_group[d_group['question'] == q]
            q_pred = p_group[p_group['question'] == q]

            avg_direct = q_direct['answer_binary'].mean()
            avg_pred_option1 = q_pred['pred_option1_conf'].mean() / 100.0
            avg_pred_option2 = q_pred['pred_option2_conf'].mean() / 100.0

            gap_option1 = avg_direct - avg_pred_option1
            gap_option2 = (1 - avg_direct) - avg_pred_option2

            # Surprise gap is the absolute difference
            surprise_gap = abs(gap_option1 - gap_option2)

            # Determine SP answer based on gap
            if dataset in ['cais/hle', 'kyssen/predict-the-futurebench-cutoff-June25']:
                sp_answer = 'yes' if gap_option1 > gap_option2 else 'no'
            else:
                sp_answer = 'true' if gap_option1 > gap_option2 else 'false'

            true_answer = q_direct['true_answer'].iloc[0]
            sp_correct = (sp_answer == true_answer)

            spa_data.append({
                'dataset': dataset,
                'model': model,
                'temperature': temp,
                'question': q,
                'surprise_gap': surprise_gap,
                'spa_correct': sp_correct,
                'avg_direct': avg_direct,
                'avg_pred_option1': avg_pred_option1,
                'gap_option1': gap_option1,
                'gap_option2': gap_option2
            })

spa_df = pd.DataFrame(spa_data)
print(f"Total SPA questions: {len(spa_df)}")

if len(spa_df) > 0:
    # Create binned version for line plot with bootstrap CIs
    spa_df['gap_bin'] = pd.cut(spa_df['surprise_gap'], bins=20)

    gap_accuracy_list = []
    for (dataset, gap_bin), group in spa_df.groupby(['dataset', 'gap_bin'], observed=True):
        if len(group) >= MIN_SAMPLES_PER_BIN:
            correct_array = group['spa_correct'].values.astype(float)
            mean_acc, lower_ci, upper_ci = parametric_ci(correct_array)
            mean_gap = group['surprise_gap'].mean()

            gap_accuracy_list.append({
                'dataset': dataset,
                'gap_bin': gap_bin,
                'surprise_gap': mean_gap,
                'spa_accuracy': mean_acc,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'n_samples': len(group)
            })

    gap_accuracy = pd.DataFrame(gap_accuracy_list)

    if len(gap_accuracy) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))

        for dataset in sorted(gap_accuracy['dataset'].unique()):
            dataset_data = gap_accuracy[gap_accuracy['dataset'] == dataset]
            dataset_short = dataset.split('/')[-1]
            color = dataset_palette.get(dataset, 'gray')

            ax.plot(dataset_data['surprise_gap'], dataset_data['spa_accuracy'] * 100,
                    marker='o', linewidth=2, markersize=8,
                    label=dataset_short, color=color)

            # Add error bars
            ax.fill_between(
                dataset_data['surprise_gap'],
                dataset_data['lower_ci'] * 100,
                dataset_data['upper_ci'] * 100,
                alpha=0.2,
                color=color
            )

        ax.set_xlabel('Surprise Gap', fontsize=13, fontweight='bold')
        ax.set_ylabel('SPA Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'Surprisingly Popular Answer Accuracy vs Surprise Gap\n(with 95% Confidence Intervals, SEM)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(title='Dataset', fontsize=11, title_fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        filename = f"{results_dir}/surprise_gap_vs_accuracy_with_ci.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.close()
    else:
        print("Not enough data for surprise gap plot")
else:
    print("No SPA data available")

# ============================================================================
# 2. HISTOGRAM OF SURPRISE GAPS
# ============================================================================
print("\n" + "="*60)
print("GRAPH 2: HISTOGRAM OF SURPRISE GAPS")
print("="*60)

if len(spa_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))

    for dataset in sorted(spa_df['dataset'].unique()):
        dataset_data = spa_df[spa_df['dataset'] == dataset]
        dataset_short = dataset.split('/')[-1]
        ax.hist(dataset_data['surprise_gap'], bins=30, alpha=0.6,
                label=dataset_short, color=dataset_palette.get(dataset, 'gray'),
                edgecolor='white', linewidth=1.5)

    ax.set_xlabel('Surprise Gap', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Surprise Gaps\n(by Dataset)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(title='Dataset', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    filename = f"{results_dir}/surprise_gap_histogram.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()
else:
    print("No SPA data available for histogram")

# ============================================================================
# 3. MARGIN OF VICTORY VS MAJORITY VOTE ACCURACY with Bootstrap CIs
# ============================================================================
print("\n" + "="*60)
print("GRAPH 3: MARGIN OF VICTORY VS MAJORITY VOTE ACCURACY")
print("="*60)

majority_data = []

for (dataset, model, temp), group in direct_df.groupby(['config_dataset_name', 'config_model_id', 'config_temperature']):
    for question in group['question'].unique():
        q_data = group[group['question'] == question]

        # Get answer counts
        answer_counts = q_data['extracted_answer'].value_counts()

        if len(answer_counts) >= 2:
            top_two = answer_counts.nlargest(2)
            margin_of_victory = (top_two.iloc[0] - top_two.iloc[1]) / len(q_data)
        elif len(answer_counts) == 1:
            margin_of_victory = 1.0
        else:
            continue

        # Get majority answer
        majority_answer = answer_counts.idxmax()
        true_answer = q_data['true_answer'].iloc[0]
        majority_correct = (majority_answer == true_answer)

        majority_data.append({
            'dataset': dataset,
            'model': model,
            'temperature': temp,
            'question': question,
            'margin_of_victory': margin_of_victory,
            'majority_correct': majority_correct
        })

majority_df = pd.DataFrame(majority_data)
print(f"Total majority vote questions: {len(majority_df)}")

if len(majority_df) > 0:
    # Create binned version with bootstrap CIs
    majority_df['margin_bin'] = pd.cut(majority_df['margin_of_victory'], bins=20)

    margin_accuracy_list = []
    for (dataset, margin_bin), group in majority_df.groupby(['dataset', 'margin_bin'], observed=True):
        if len(group) >= MIN_SAMPLES_PER_BIN:
            correct_array = group['majority_correct'].values.astype(float)
            mean_acc, lower_ci, upper_ci = parametric_ci(correct_array)
            mean_margin = group['margin_of_victory'].mean()

            margin_accuracy_list.append({
                'dataset': dataset,
                'margin_bin': margin_bin,
                'margin_of_victory': mean_margin,
                'majority_accuracy': mean_acc,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'n_samples': len(group)
            })

    margin_accuracy = pd.DataFrame(margin_accuracy_list)

    if len(margin_accuracy) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))

        for dataset in sorted(margin_accuracy['dataset'].unique()):
            dataset_data = margin_accuracy[margin_accuracy['dataset'] == dataset]
            dataset_short = dataset.split('/')[-1]
            color = dataset_palette.get(dataset, 'gray')

            ax.plot(dataset_data['margin_of_victory'], dataset_data['majority_accuracy'] * 100,
                    marker='o', linewidth=2, markersize=8,
                    label=dataset_short, color=color)

            # Add error bars
            ax.fill_between(
                dataset_data['margin_of_victory'],
                dataset_data['lower_ci'] * 100,
                dataset_data['upper_ci'] * 100,
                alpha=0.2,
                color=color
            )

        ax.set_xlabel('Margin of Victory', fontsize=13, fontweight='bold')
        ax.set_ylabel('Majority Vote Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'Majority Vote Accuracy vs Margin of Victory\n(with 95% Confidence Intervals, SEM)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(title='Dataset', fontsize=11, title_fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        filename = f"{results_dir}/margin_of_victory_vs_accuracy_with_ci.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.close()
    else:
        print("Not enough data for margin of victory plot")
else:
    print("No majority vote data available")

# ============================================================================
# 4. CONFIDENCE CALIBRATION PLOTS (One per dataset) with Bootstrap CIs
# ============================================================================
print("\n" + "="*60)
print("GRAPH 4: CONFIDENCE CALIBRATION PLOTS")
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

    # Create one plot per dataset
    for dataset in sorted(conf_df['config_dataset_name'].unique()):
        dataset_data = conf_df[conf_df['config_dataset_name'] == dataset]
        dataset_short = dataset.split('/')[-1]

        fig, ax = plt.subplots(figsize=(10, 10))

        for model in sorted(dataset_data['config_model_id'].unique()):
            model_data = dataset_data[dataset_data['config_model_id'] == model].copy()
            model_short = model.split('.')[-1].split(':')[0]

            # Bin confidence scores
            model_data.loc[:, 'conf_bin'] = pd.cut(model_data['confidence'], bins=N_BINS)

            binned_list = []
            for conf_bin, group in model_data.groupby('conf_bin', observed=True):
                if len(group) >= MIN_SAMPLES_PER_BIN:
                    correct_array = group['correct'].values.astype(float)
                    mean_acc, lower_ci, upper_ci = parametric_ci(correct_array)
                    mean_conf = group['confidence'].mean() / 100.0

                    binned_list.append({
                        'conf_bin': conf_bin,
                        'confidence': mean_conf,
                        'accuracy': mean_acc,
                        'lower_ci': lower_ci,
                        'upper_ci': upper_ci,
                        'n_samples': len(group)
                    })

            if binned_list:
                binned = pd.DataFrame(binned_list)
                color = model_palette.get(model, 'gray')

                ax.plot(binned['confidence'], binned['accuracy'],
                        marker='o', linewidth=2, markersize=8,
                        label=model_short, color=color)

                # Add error bars
                ax.fill_between(
                    binned['confidence'],
                    binned['lower_ci'],
                    binned['upper_ci'],
                    alpha=0.2,
                    color=color
                )

        # Add perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')

        ax.set_xlabel('Confidence', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax.set_title(f'Confidence Calibration - {dataset_short.upper()}\n(with 95% Confidence Intervals, SEM)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(title='Model', fontsize=9, title_fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        filename = f"{results_dir}/confidence_calibration_{dataset.replace('/', '_')}_with_ci.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.close()
else:
    print("No confidence data available")

# ============================================================================
# 5. AGREEMENT FRACTION VS CONFIDENCE (Aggregate across datasets) with Bootstrap CIs
# ============================================================================
print("\n" + "="*60)
print("GRAPH 5: AGREEMENT FRACTION VS CONFIDENCE")
print("="*60)

if len(conf_df) > 0:
    agreement_data = []

    for (dataset, model, temp, question), group in conf_df.groupby(['config_dataset_name', 'config_model_id', 'config_temperature', 'question']):
        # Get the most popular answer
        answer_counts = group['extracted_answer'].value_counts()
        if len(answer_counts) == 0:
            continue

        most_popular = answer_counts.idxmax()

        # Get mean confidence of those who gave the most popular answer
        popular_responses = group[group['extracted_answer'] == most_popular]
        if len(popular_responses) == 0:
            continue

        mean_confidence = popular_responses['confidence'].mean()

        # Agreement fraction = fraction that agree with most popular answer
        agreement_fraction = len(popular_responses) / len(group)

        agreement_data.append({
            'dataset': dataset,
            'model': model,
            'temperature': temp,
            'question': question,
            'mean_confidence': mean_confidence / 100,  # Normalize to 0-1
            'agreement_fraction': agreement_fraction
        })

    agreement_df = pd.DataFrame(agreement_data)
    print(f"Total agreement questions: {len(agreement_df)}")

    if len(agreement_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 10))

        for model in sorted(agreement_df['model'].unique()):
            model_data = agreement_df[agreement_df['model'] == model].copy()
            model_short = model.split('.')[-1].split(':')[0]

            # Bin by confidence
            model_data.loc[:, 'conf_bin'] = pd.cut(model_data['mean_confidence'], bins=N_BINS)

            binned_list = []
            for conf_bin, group in model_data.groupby('conf_bin', observed=True):
                if len(group) >= MIN_SAMPLES_PER_BIN:
                    agreement_array = group['agreement_fraction'].values
                    mean_agree, lower_ci, upper_ci = parametric_ci(agreement_array)
                    mean_conf = group['mean_confidence'].mean()

                    binned_list.append({
                        'conf_bin': conf_bin,
                        'mean_confidence': mean_conf,
                        'agreement_fraction': mean_agree,
                        'lower_ci': lower_ci,
                        'upper_ci': upper_ci,
                        'n_samples': len(group)
                    })

            if binned_list:
                binned = pd.DataFrame(binned_list)
                color = model_palette.get(model, 'gray')

                ax.plot(binned['mean_confidence'], binned['agreement_fraction'],
                        marker='o', linewidth=2, markersize=8,
                        label=model_short, color=color)

                # Add error bars
                ax.fill_between(
                    binned['mean_confidence'],
                    binned['lower_ci'],
                    binned['upper_ci'],
                    alpha=0.2,
                    color=color
                )

        # Add perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')

        ax.set_xlabel('Mean Confidence (Most Popular Answer)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Agreement Fraction', fontsize=13, fontweight='bold')
        ax.set_title(f'Agreement Fraction vs Confidence\n(with 95% Confidence Intervals, SEM)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(title='Model', fontsize=9, title_fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        filename = f"{results_dir}/agreement_vs_confidence_with_ci.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.close()
    else:
        print("Not enough agreement data for plot")
else:
    print("No confidence data available for agreement analysis")

# ============================================================================
# 6. PREDICTED POPULARITY VS AGREEMENT FRACTION (One per dataset) with Bootstrap CIs
# ============================================================================
print("\n" + "="*60)
print("GRAPH 6: PREDICTED POPULARITY VS AGREEMENT FRACTION")
print("="*60)

if len(spa_df) > 0:
    pred_agreement_data = []

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

            # Agreement fraction with correct answer
            agreement_with_correct = (q_direct['extracted_answer'] == true_answer).mean()

            # Predicted popularity of correct answer
            avg_pred_option1 = q_pred['pred_option1_conf'].mean() / 100.0

            # Determine which option corresponds to the correct answer
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

    # Create one plot per dataset
    for dataset in sorted(pred_agreement_df['dataset'].unique()):
        dataset_data = pred_agreement_df[pred_agreement_df['dataset'] == dataset]
        dataset_short = dataset.split('/')[-1]

        fig, ax = plt.subplots(figsize=(10, 10))

        for model in sorted(dataset_data['model'].unique()):
            model_data = dataset_data[dataset_data['model'] == model].copy()
            model_short = model.split('.')[-1].split(':')[0]

            # Bin by predicted popularity
            model_data.loc[:, 'pred_bin'] = pd.cut(model_data['predicted_popularity'], bins=N_BINS)

            binned_list = []
            for pred_bin, group in model_data.groupby('pred_bin', observed=True):
                if len(group) >= MIN_SAMPLES_PER_BIN:
                    agreement_array = group['agreement_with_correct'].values
                    mean_agree, lower_ci, upper_ci = parametric_ci(agreement_array)
                    mean_pred = group['predicted_popularity'].mean()

                    binned_list.append({
                        'pred_bin': pred_bin,
                        'predicted_popularity': mean_pred,
                        'agreement_with_correct': mean_agree,
                        'lower_ci': lower_ci,
                        'upper_ci': upper_ci,
                        'n_samples': len(group)
                    })

            if binned_list:
                binned = pd.DataFrame(binned_list)
                color = model_palette.get(model, 'gray')

                ax.plot(binned['predicted_popularity'], binned['agreement_with_correct'],
                        marker='o', linewidth=2, markersize=8,
                        label=model_short, color=color)

                # Add error bars
                ax.fill_between(
                    binned['predicted_popularity'],
                    binned['lower_ci'],
                    binned['upper_ci'],
                    alpha=0.2,
                    color=color
                )

        # Add perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')

        ax.set_xlabel('Predicted Popularity (Correct Answer)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Agreement Fraction (Correct Answer)', fontsize=13, fontweight='bold')
        ax.set_title(f'Predicted Popularity vs Agreement - {dataset_short.upper()}\n(with 95% Confidence Intervals, SEM)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(title='Model', fontsize=9, title_fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        filename = f"{results_dir}/predicted_popularity_vs_agreement_{dataset.replace('/', '_')}_with_ci.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.close()
else:
    print("No prediction data available")

print("\n" + "="*60)
print("ALL VISUALIZATIONS COMPLETE")
print("="*60)
print(f"All plots saved to: {results_dir}")
