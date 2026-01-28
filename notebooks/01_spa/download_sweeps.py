import os
import re
import pandas as pd
import numpy as np
import wandb
from src.analyze import download_wandb_project_runs_configs, download_wandb_sweep_runs_responses, setup_notebook_dir, fix_com2sense_labels
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)    # for reproducibility

REFRESH = False
N_BOOTSTRAP = 1000  # Number of bootstrap samples
CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals

# Configuration and data loading
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir, results_dir = setup_notebook_dir(script_dir, refresh=REFRESH)
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

# Fix com2sense labels before filtering
print(f"\n{'='*60}")
print(f"FIXING COM2SENSE LABELS")
print(f"{'='*60}")
responses_df = fix_com2sense_labels(responses_df, data_dir)

# Filter unclear answers
responses_df = responses_df[responses_df['extracted_answer'] != 'unclear']

# Bootstrap function - VECTORIZED for speed
def bootstrap_accuracy(correct_array, n_bootstrap=N_BOOTSTRAP, confidence_level=CONFIDENCE_LEVEL):
    """
    Fast vectorized bootstrap confidence intervals for accuracy
    
    Args:
        correct_array: Binary array of correct (1) / incorrect (0)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
    
    Returns:
        mean, lower_ci, upper_ci
    """
    n = len(correct_array)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # Vectorized resampling - much faster!
    bootstrap_indices = np.random.randint(0, n, size=(n_bootstrap, n))
    bootstrap_samples = correct_array[bootstrap_indices]
    bootstrap_means = bootstrap_samples.mean(axis=1)
    
    mean_acc = correct_array.mean()
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_ci = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_ci = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return mean_acc, lower_ci, upper_ci


def process_answers(df, response_type, extract_confidence=False, extract_prediction=False):
    """Process answers and calculate individual and majority vote accuracy with bootstrapping"""
    df = df[df['response_type'] == response_type].copy()
    
    if len(df) == 0:
        print(f"No data found for response_type: {response_type}")
        return None, None, None
    
    # Extract confidence if needed
    if extract_confidence:
        def extract(response):
            if pd.isna(response) or not isinstance(response, str):
                return None, None
            
            ans = re.search(r'Answer:\s*(\w+)', response, re.IGNORECASE)
            conf = re.search(r'Confidence:\s*([\d.]+)', response, re.IGNORECASE)
            
            if ans and conf:
                answer = ans.group(1)
                confidence = float(conf.group(1))
                return (answer, confidence)
            
            return None, None
        
        extracted = df['model_response'].apply(extract)
        df['extracted_answer'] = extracted.apply(lambda x: x[0])
        df['confidence'] = extracted.apply(lambda x: x[1])
        df = df.dropna(subset=['extracted_answer', 'confidence'])
        
        if len(df) == 0:
            print(f"No valid data after extraction for response_type: {response_type}")
            return None, None, None
    
    # Extract prediction fractions if needed
    if extract_prediction:
        def extract_pred(response, dataset):
            if pd.isna(response) or not isinstance(response, str):
                return None, None, None
            
            if dataset == 'cais/hle' or dataset == 'kyssen/predict-the-futurebench-cutoff-June25':
                yes_match = re.search(r'YES:\s*(\d+)', response, re.IGNORECASE)
                no_match = re.search(r'NO:\s*(\d+)', response, re.IGNORECASE)
                if yes_match and no_match:
                    yes_conf = int(yes_match.group(1))
                    no_conf = int(no_match.group(1))
                    answer = 'yes' if yes_conf > no_conf else 'no'
                    confidence = max(yes_conf, no_conf)
                    return (answer, confidence, yes_conf if answer == 'yes' else no_conf)
            else:
                true_match = re.search(r'TRUE:\s*(\d+)', response, re.IGNORECASE)
                false_match = re.search(r'FALSE:\s*(\d+)', response, re.IGNORECASE)
                if true_match and false_match:
                    true_conf = int(true_match.group(1))
                    false_conf = int(false_match.group(1))
                    answer = 'true' if true_conf > false_conf else 'false'
                    confidence = max(true_conf, false_conf)
                    return (answer, confidence, true_conf if answer == 'true' else false_conf)
            
            return None, None, None
        
        extracted = df.apply(lambda row: extract_pred(row['model_response'], row['config_dataset_name']), axis=1)
        df['extracted_answer'] = extracted.apply(lambda x: x[0])
        df['confidence'] = extracted.apply(lambda x: x[1])
        df = df.dropna(subset=['extracted_answer', 'confidence'])
        
        if len(df) == 0:
            print(f"No valid data after extraction for response_type: {response_type}")
            return None, None, None
    
    # Normalize all answers to lowercase strings for consistent comparison
    # Handle extracted_answer - convert to lowercase string
    df['extracted_answer'] = df['extracted_answer'].apply(
        lambda x: str(x).lower() if pd.notna(x) else None
    )

    # Handle true_answer - convert to lowercase string
    df['true_answer'] = df['true_answer'].apply(
        lambda x: str(x).lower() if pd.notna(x) else None
    )

    # Remove any None values that were created
    df = df.dropna(subset=['extracted_answer', 'true_answer'])
    
    # Calculate correctness (both are already lowercase strings)
    df['correct'] = (df['extracted_answer'].str.strip() == df['true_answer'].str.strip())
    
    # Individual accuracy with bootstrapping
    individual_results = []
    for (dataset, model, temp), group in df.groupby(['config_dataset_name', 'config_model_id', 'config_temperature']):
        correct_array = group['correct'].values.astype(float)
        mean_acc, lower_ci, upper_ci = bootstrap_accuracy(correct_array)
        individual_results.append({
            'dataset': dataset,
            'model': model,
            'temperature': temp,
            'accuracy': mean_acc * 100,
            'lower_ci': lower_ci * 100,
            'upper_ci': upper_ci * 100,
            'num_responses': len(correct_array)
        })
    
    individual = pd.DataFrame(individual_results)
    
    # Majority vote accuracy with bootstrapping
    def get_majority_correct_for_bootstrap(x, use_confidence=False):
        """Get majority answer for a group of responses"""
        if use_confidence and 'confidence' in x.columns:
            weighted_votes = x.groupby('extracted_answer')['confidence'].sum()
        else:
            weighted_votes = x['extracted_answer'].value_counts()
        
        if len(weighted_votes) == 0:
            return None
        
        majority_answer = weighted_votes.idxmax()
        return majority_answer == x['true_answer'].iloc[0]
    
    # Majority vote accuracy with bootstrapping - VECTORIZED
    majority_results = []
    for (dataset, model, temp), group in df.groupby(['config_dataset_name', 'config_model_id', 'config_temperature']):
        # Get unique questions and their majority votes
        questions = group['question'].unique()
        n_questions = len(questions)
        
        # Pre-compute majority vote correctness for each question
        question_correct = {}
        for q in questions:
            q_data = group[group['question'] == q]
            is_correct = get_majority_correct_for_bootstrap(
                q_data, 
                use_confidence=(extract_confidence or extract_prediction)
            )
            if is_correct is not None:
                question_correct[q] = is_correct
        
        # Convert to array for vectorized operations
        correct_array = np.array([question_correct[q] for q in questions if q in question_correct])
        
        if len(correct_array) > 0:
            # Vectorized bootstrap
            bootstrap_indices = np.random.randint(0, len(correct_array), size=(N_BOOTSTRAP, len(correct_array)))
            bootstrap_samples = correct_array[bootstrap_indices]
            bootstrap_accs = bootstrap_samples.mean(axis=1)
            
            mean_acc = bootstrap_accs.mean()
            alpha = 1 - CONFIDENCE_LEVEL
            lower_ci = np.percentile(bootstrap_accs, 100 * alpha / 2)
            upper_ci = np.percentile(bootstrap_accs, 100 * (1 - alpha / 2))
            
            majority_results.append({
                'dataset': dataset,
                'model': model,
                'temperature': temp,
                'accuracy': mean_acc * 100,
                'lower_ci': lower_ci * 100,
                'upper_ci': upper_ci * 100
            })
    
    majority_agg = pd.DataFrame(majority_results)
    
    # Comparison
    comparison = individual.merge(
        majority_agg, 
        on=['dataset', 'model', 'temperature'], 
        suffixes=('_individual', '_majority')
    )
    comparison['improvement'] = comparison['accuracy_majority'] - comparison['accuracy_individual']
    
    return individual, majority_agg, comparison


# Process direct answers
print("=" * 60, "\nDIRECT ANSWERS\n", "=" * 60)
direct_ind, direct_maj, direct_comp = process_answers(responses_df, 'direct_answer')
if direct_ind is not None:
    print("\nIndividual:\n", direct_ind)
    print("\nMajority Vote:\n", direct_maj)
    print("\nComparison:\n", direct_comp[['dataset', 'model', 'temperature', 'accuracy_individual', 'accuracy_majority', 'improvement']])

# Process confidence-weighted answers
print("\n" + "=" * 60, "\nCONFIDENCE-WEIGHTED MAJORITY VOTE\n", "=" * 60)
result = process_answers(responses_df, 'confidence', extract_confidence=True)
if result[0] is not None:
    conf_ind, conf_maj, _ = result
    print("\n", conf_maj)

# Process prediction answers
print("\n" + "=" * 60, "\nPREDICTION-WEIGHTED MAJORITY VOTE\n", "=" * 60)
result = process_answers(responses_df, 'prediction', extract_prediction=True)
if result[0] is not None:
    pred_ind, pred_maj, _ = result
    print("\n", pred_maj)


# Compute highest confidence selection with bootstrapping
print("\n" + "=" * 60, "\nHIGHEST CONFIDENCE SELECTION\n", "=" * 60)

def compute_highest_confidence(df):
    """For each question, select the answer with the highest confidence - VECTORIZED"""
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
    
    # Normalize all answers to lowercase strings for consistent comparison
    conf_df['extracted_answer'] = conf_df['extracted_answer'].apply(
        lambda x: str(x).lower() if pd.notna(x) else None
    )
    conf_df['true_answer'] = conf_df['true_answer'].apply(
        lambda x: str(x).lower() if pd.notna(x) else None
    )
    conf_df = conf_df.dropna(subset=['extracted_answer', 'true_answer'])
    
    # Vectorized bootstrap by question
    results = []
    for (dataset, model, temp), group in conf_df.groupby(['config_dataset_name', 'config_model_id', 'config_temperature']):
        questions = group['question'].unique()
        
        # Pre-compute correctness for each question
        question_correct = {}
        for q in questions:
            q_data = group[group['question'] == q]
            max_idx = q_data['confidence'].idxmax()
            selected_answer = q_data.loc[max_idx, 'extracted_answer']
            true_answer = q_data['true_answer'].iloc[0]
            question_correct[q] = (selected_answer == true_answer)
        
        # Convert to array
        correct_array = np.array([question_correct[q] for q in questions])
        
        # Vectorized bootstrap
        bootstrap_indices = np.random.randint(0, len(correct_array), size=(N_BOOTSTRAP, len(correct_array)))
        bootstrap_samples = correct_array[bootstrap_indices]
        bootstrap_accs = bootstrap_samples.mean(axis=1)
        
        mean_acc = bootstrap_accs.mean()
        alpha = 1 - CONFIDENCE_LEVEL
        lower_ci = np.percentile(bootstrap_accs, 100 * alpha / 2)
        upper_ci = np.percentile(bootstrap_accs, 100 * (1 - alpha / 2))
        
        results.append({
            'dataset': dataset,
            'model': model,
            'temperature': temp,
            'accuracy': mean_acc * 100,
            'lower_ci': lower_ci * 100,
            'upper_ci': upper_ci * 100
        })
    
    return pd.DataFrame(results)

highest_conf_result = compute_highest_confidence(responses_df)
if highest_conf_result is not None:
    print("\n", highest_conf_result)


# Compute Surprisingly Popular Answer with bootstrapping
print("\n" + "=" * 60, "\nSURPRISINGLY POPULAR ANSWER\n", "=" * 60)

def compute_surprisingly_popular(df):
    """Compute surprisingly popular answer - VECTORIZED"""
    
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
    
    # Normalize direct_df answers to lowercase strings
    direct_df['extracted_answer'] = direct_df['extracted_answer'].apply(
        lambda x: str(x).lower() if pd.notna(x) else None
    )
    direct_df['true_answer'] = direct_df['true_answer'].apply(
        lambda x: str(x).lower() if pd.notna(x) else None
    )
    direct_df = direct_df.dropna(subset=['extracted_answer', 'true_answer'])

    # Convert direct answers to binary
    direct_df['answer_binary'] = 0

    for dataset in ['cais/hle', 'kyssen/predict-the-futurebench-cutoff-June25']:
        dataset_mask = direct_df['config_dataset_name'] == dataset
        direct_df.loc[dataset_mask & direct_df['extracted_answer'].isin(['yes']), 'answer_binary'] = 1

    for dataset in ['google/boolq', 'tasksource/com2sense']:
        dataset_mask = direct_df['config_dataset_name'] == dataset
        direct_df.loc[dataset_mask & direct_df['extracted_answer'].isin(['true']), 'answer_binary'] = 1
    
    # Vectorized bootstrap by question
    results = []
    for (dataset, model, temp), d_group in direct_df.groupby(['config_dataset_name', 'config_model_id', 'config_temperature']):
        p_group = pred_df[(pred_df['config_dataset_name'] == dataset) & 
                          (pred_df['config_model_id'] == model) & 
                          (pred_df['config_temperature'] == temp)]
        
        if len(p_group) == 0:
            continue
        
        # Get common questions
        common_questions = list(set(d_group['question'].unique()) & set(p_group['question'].unique()))
        
        if len(common_questions) == 0:
            continue
        
        # Pre-compute SP correctness for each question
        question_correct = {}
        for q in common_questions:
            q_direct = d_group[d_group['question'] == q]
            q_pred = p_group[p_group['question'] == q]
            
            avg_direct = q_direct['answer_binary'].mean()
            avg_pred_true = q_pred['pred_true_conf'].mean()
            avg_pred_false = q_pred['pred_false_conf'].mean()
            
            gap_true = avg_direct - (avg_pred_true / 100)
            gap_false = (1 - avg_direct) - (avg_pred_false / 100)
            
            # Determine SP answer based on gap - return as string matching dataset format
            if dataset in ['cais/hle', 'kyssen/predict-the-futurebench-cutoff-June25']:
                sp_answer = 'yes' if gap_true > gap_false else 'no'
            else:  # google/boolq, tasksource/com2sense
                sp_answer = 'true' if gap_true > gap_false else 'false'

            true_answer = q_direct['true_answer'].iloc[0]

            question_correct[q] = (sp_answer == true_answer)
        
        # Convert to array
        correct_array = np.array([question_correct[q] for q in common_questions])
        
        # Vectorized bootstrap
        bootstrap_indices = np.random.randint(0, len(correct_array), size=(N_BOOTSTRAP, len(correct_array)))
        bootstrap_samples = correct_array[bootstrap_indices]
        bootstrap_accs = bootstrap_samples.mean(axis=1)
        
        mean_acc = bootstrap_accs.mean()
        alpha = 1 - CONFIDENCE_LEVEL
        lower_ci = np.percentile(bootstrap_accs, 100 * alpha / 2)
        upper_ci = np.percentile(bootstrap_accs, 100 * (1 - alpha / 2))
        
        results.append({
            'dataset': dataset,
            'model': model,
            'temperature': temp,
            'accuracy': mean_acc * 100,
            'lower_ci': lower_ci * 100,
            'upper_ci': upper_ci * 100
        })
    
    return pd.DataFrame(results)

sp_result = compute_surprisingly_popular(responses_df)
if sp_result is not None:
    print("\n", sp_result)


# Compute Inverse Surprisingly Popular Answer with bootstrapping
print("\n" + "=" * 60, "\nPROXY ISP (INVERSE SURPRISINGLY POPULAR)\n", "=" * 60)

def compute_inverse_surprisingly_popular(df):
    """Compute inverse surprisingly popular answer (ISP) using paired answer/prediction samples."""

    # Only use SPA experiment data (needs predictions)
    df = df[df['config_experiment_type'] == 'surprisingly_popular'].copy()

    direct_df = df[df['response_type'] == 'direct_answer'].copy()
    if len(direct_df) == 0:
        print("No direct_answer data found")
        return None

    pred_df = df[df['response_type'] == 'prediction'].copy()
    if len(pred_df) == 0:
        print("No prediction data found")
        return None

    # Normalize direct answers
    direct_df['extracted_answer'] = direct_df['extracted_answer'].apply(
        lambda x: str(x).lower().strip() if pd.notna(x) else None
    )
    direct_df['true_answer'] = direct_df['true_answer'].apply(
        lambda x: str(x).lower().strip() if pd.notna(x) else None
    )
    direct_df = direct_df.dropna(subset=['extracted_answer', 'true_answer'])
    direct_df = direct_df[direct_df['extracted_answer'].isin(['yes', 'no', 'true', 'false'])]

    yes_no_datasets = {'cais/hle', 'kyssen/predict-the-futurebench-cutoff-June25'}
    true_false_datasets = {'google/boolq', 'tasksource/com2sense'}

    # Extract predictions (support percent or probability formats)
    def extract_predictions(row):
        response = row.get('model_response')
        if not isinstance(response, str) or response.strip() == "":
            response = row.get('extracted_answer')
        if not isinstance(response, str):
            return None, None

        dataset = row.get('config_dataset_name')
        if dataset in yes_no_datasets:
            m1 = re.search(r'YES:?\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
            m2 = re.search(r'NO:?\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
        elif dataset in true_false_datasets:
            m1 = re.search(r'TRUE:?\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
            m2 = re.search(r'FALSE:?\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
        else:
            m1 = re.search(r'YES:?\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
            m2 = re.search(r'NO:?\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
            if not (m1 and m2):
                m1 = re.search(r'TRUE:?\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
                m2 = re.search(r'FALSE:?\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)

        if not (m1 and m2):
            return None, None

        v1 = float(m1.group(1))
        v2 = float(m2.group(1))
        if v1 + v2 > 2.0:  # likely percentages
            v1 /= 100.0
            v2 /= 100.0

        return v1, v2

    extracted_pred = pred_df.apply(extract_predictions, axis=1)
    pred_df['pred_option1'] = extracted_pred.apply(lambda x: x[0])
    pred_df['pred_option2'] = extracted_pred.apply(lambda x: x[1])
    pred_df = pred_df.dropna(subset=['pred_option1', 'pred_option2'])

    # Pair direct answers with predictions by run/question/sample index
    key_cols = ['run_id', 'question_idx', 'response_idx']
    # run_id is a string; only coerce numeric indices
    direct_df['question_idx'] = pd.to_numeric(direct_df['question_idx'], errors='coerce')
    direct_df['response_idx'] = pd.to_numeric(direct_df['response_idx'], errors='coerce')
    pred_df['question_idx'] = pd.to_numeric(pred_df['question_idx'], errors='coerce')
    pred_df['response_idx'] = pd.to_numeric(pred_df['response_idx'], errors='coerce')

    direct_df = direct_df.dropna(subset=key_cols)
    pred_df = pred_df.dropna(subset=key_cols)

    direct_df['question_idx'] = direct_df['question_idx'].astype(int)
    direct_df['response_idx'] = direct_df['response_idx'].astype(int)
    pred_df['question_idx'] = pred_df['question_idx'].astype(int)
    pred_df['response_idx'] = pred_df['response_idx'].astype(int)

    # Pair by response_idx to approximate per-agent answer + prediction linkage
    paired = direct_df.merge(
        pred_df[key_cols + ['pred_option1', 'pred_option2']],
        on=key_cols,
        how='inner'
    )

    results = []
    for (dataset, model, temp), group in paired.groupby(['config_dataset_name', 'config_model_id', 'config_temperature']):
        if len(group) == 0:
            continue

        option1 = 'yes' if dataset in yes_no_datasets else 'true'
        option2 = 'no' if dataset in yes_no_datasets else 'false'

        common_questions = group['question'].unique()
        if len(common_questions) == 0:
            continue

        question_correct = {}
        for q in common_questions:
            q_group = group[group['question'] == q]

            n1 = (q_group['extracted_answer'] == option1).sum()
            n2 = (q_group['extracted_answer'] == option2).sum()
            if n1 + n2 == 0:
                continue

            overall_p1 = q_group['pred_option1'].mean()
            overall_p2 = q_group['pred_option2'].mean()

            p1_given_1 = q_group[q_group['extracted_answer'] == option1]['pred_option1'].mean()
            p1_given_2 = q_group[q_group['extracted_answer'] == option2]['pred_option1'].mean()
            p2_given_1 = q_group[q_group['extracted_answer'] == option1]['pred_option2'].mean()
            p2_given_2 = q_group[q_group['extracted_answer'] == option2]['pred_option2'].mean()

            if pd.isna(p1_given_1):
                p1_given_1 = overall_p1
            if pd.isna(p1_given_2):
                p1_given_2 = overall_p1
            if pd.isna(p2_given_1):
                p2_given_1 = overall_p2
            if pd.isna(p2_given_2):
                p2_given_2 = overall_p2

            # ISP advantage for binary case (K=2)
            adv_opt1 = n1 - (n1 * p1_given_2 + n2 * p1_given_1)
            adv_opt2 = n2 - (n1 * p2_given_2 + n2 * p2_given_1)

            isp_answer = option1 if adv_opt1 > adv_opt2 else option2
            true_answer = q_group['true_answer'].iloc[0]
            question_correct[q] = (isp_answer == true_answer)

        correct_array = np.array([question_correct[q] for q in common_questions if q in question_correct])
        if len(correct_array) == 0:
            continue

        bootstrap_indices = np.random.randint(0, len(correct_array), size=(N_BOOTSTRAP, len(correct_array)))
        bootstrap_samples = correct_array[bootstrap_indices]
        bootstrap_accs = bootstrap_samples.mean(axis=1)

        mean_acc = bootstrap_accs.mean()
        alpha = 1 - CONFIDENCE_LEVEL
        lower_ci = np.percentile(bootstrap_accs, 100 * alpha / 2)
        upper_ci = np.percentile(bootstrap_accs, 100 * (1 - alpha / 2))

        results.append({
            'dataset': dataset,
            'model': model,
            'temperature': temp,
            'accuracy': mean_acc * 100,
            'lower_ci': lower_ci * 100,
            'upper_ci': upper_ci * 100
        })

    return pd.DataFrame(results)

isp_result = compute_inverse_surprisingly_popular(responses_df)
if isp_result is not None:
    print("\n", isp_result)


# Create visualizations with seaborn
print("\n" + "=" * 60, "\nCREATING VISUALIZATIONS\n", "=" * 60)

# Set style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.0)
sns.set_palette("husl")

def prepare_visualization_data():
    """Combine results from all methods"""
    
    individual_results = direct_ind[['dataset', 'model', 'temperature', 'accuracy', 'lower_ci', 'upper_ci']].copy()
    individual_results['method'] = 'Individual Avg'
    
    direct_results = direct_maj[['dataset', 'model', 'temperature', 'accuracy', 'lower_ci', 'upper_ci']].copy()
    direct_results['method'] = 'Direct Majority'
    
    highest_conf_results = highest_conf_result[['dataset', 'model', 'temperature', 'accuracy', 'lower_ci', 'upper_ci']].copy()
    highest_conf_results['method'] = 'Highest Conf'
    
    conf_results = conf_maj[['dataset', 'model', 'temperature', 'accuracy', 'lower_ci', 'upper_ci']].copy()
    conf_results['method'] = 'Conf Weighted'
    
    pred_results = pred_maj[['dataset', 'model', 'temperature', 'accuracy', 'lower_ci', 'upper_ci']].copy()
    pred_results['method'] = 'Pred Weighted'
    
    sp_results = sp_result[['dataset', 'model', 'temperature', 'accuracy', 'lower_ci', 'upper_ci']].copy()
    sp_results['method'] = 'Surp. Popular'

    results_list = [
        individual_results, direct_results, highest_conf_results,
        conf_results, pred_results, sp_results
    ]

    if isp_result is not None and len(isp_result) > 0:
        isp_results = isp_result[['dataset', 'model', 'temperature', 'accuracy', 'lower_ci', 'upper_ci']].copy()
        isp_results['method'] = 'Proxy ISP'
        results_list.append(isp_results)
    
    all_results = pd.concat(results_list, ignore_index=True)
    
    # Calculate error bars
    all_results['error_lower'] = all_results['accuracy'] - all_results['lower_ci']
    all_results['error_upper'] = all_results['upper_ci'] - all_results['accuracy']
    
    return all_results

viz_data = prepare_visualization_data()

# Define color palette
palette = {
    'Individual Avg': '#95a5a6',
    'Direct Majority': '#3498db',
    'Highest Conf': '#9b59b6',
    'Conf Weighted': '#e74c3c', 
    'Pred Weighted': '#2ecc71',
    'Surp. Popular': '#f39c12',
    'Proxy ISP': '#1abc9c'
}

methods = ['Individual Avg', 'Direct Majority', 'Highest Conf', 'Conf Weighted', 'Pred Weighted', 'Surp. Popular']
if isp_result is not None and len(isp_result) > 0:
    methods.append('Proxy ISP')
datasets = viz_data['dataset'].unique()

# Create plots for each dataset
for dataset in datasets:
    dataset_data = viz_data[viz_data['dataset'] == dataset].copy()
    
    # Create model-temperature label
    dataset_data['model_temp'] = dataset_data.apply(
        lambda row: f"{row['model'].split('.')[-1].split(':')[0]}\n(T={row['temperature']})", 
        axis=1
    )
    
    # Get unique model-temp combinations in sorted order
    models = sorted(dataset_data['model'].unique())
    temperatures = sorted(dataset_data['temperature'].dropna().unique())
    model_temp_order = []
    for model in models:
        for temp in temperatures:
            subset = dataset_data[(dataset_data['model'] == model) & 
                                 (dataset_data['temperature'] == temp)]
            if len(subset) > 0:
                model_temp_order.append(subset['model_temp'].iloc[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(14, len(model_temp_order) * 1.5), 7))
    
    # Prepare error bar data
    x_positions = []
    y_values = []
    y_errors = []
    hue_values = []
    
    # Get bar width and number of methods
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    
    for i, model_temp in enumerate(model_temp_order):
        for j, method in enumerate(methods):
            subset = dataset_data[(dataset_data['model_temp'] == model_temp) & 
                                 (dataset_data['method'] == method)]
            if len(subset) > 0:
                row = subset.iloc[0]
                x_positions.append(i + (j - n_methods/2 + 0.5) * bar_width)
                y_values.append(row['accuracy'])
                y_errors.append([row['error_lower'], row['error_upper']])
                hue_values.append(method)
    
    # Create barplot using seaborn
    sns.barplot(
        data=dataset_data,
        x='model_temp',
        y='accuracy',
        hue='method',
        order=model_temp_order,
        hue_order=methods,
        palette=palette,
        ax=ax,
        alpha=0.85,
        edgecolor='white',
        linewidth=1.5,
        errorbar=None  # We'll add custom error bars
    )
    
    # Add custom error bars
    for i, (x, y, yerr, hue) in enumerate(zip(x_positions, y_values, y_errors, hue_values)):
        ax.errorbar(
            x, y, 
            yerr=[[yerr[0]], [yerr[1]]],
            fmt='none',
            ecolor='black',
            elinewidth=1.5,
            capsize=3,
            capthick=1.5,
            alpha=0.7
        )
    
    # Add value labels on bars
    for container in ax.containers:
        # Skip error bar containers
        if isinstance(container, plt.matplotlib.container.BarContainer):
            labels = [f'{v:.1f}' if v > 0 else '' for v in container.datavalues]
            ax.bar_label(container, labels=labels, fontsize=9, fontweight='bold', padding=3)
    
    # Customize plot
    ax.set_xlabel('Model (Temperature)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    
    # Clean dataset name for title
    dataset_title = dataset.split('/')[-1].upper()
    ax.set_title(
        f'{dataset_title} - Accuracy by Method\n(with {int(CONFIDENCE_LEVEL*100)}% Bootstrap Confidence Intervals)', 
        fontsize=16, 
        fontweight='bold', 
        pad=20
    )
    
    ax.set_ylim(0, 105)
    ax.legend(
        title='Method', 
        loc='upper left', 
        fontsize=10, 
        framealpha=0.95, 
        title_fontsize=11,
        ncol=2 if len(methods) > 4 else 1
    )
    ax.tick_params(axis='x', labelsize=9, rotation=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    
    plt.tight_layout()
    
    # Save figure
    dataset_clean = dataset.replace('/', '_')
    filename = f"{results_dir}/accuracy_comparison_{dataset_clean}_bootstrap.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()


# Create a summary comparison plot across all datasets
print("\nCreating summary comparison plot...")

fig, ax = plt.subplots(figsize=(16, 8))

# Prepare data for comparison
summary_data = viz_data.copy()
summary_data['dataset_short'] = summary_data['dataset'].apply(lambda x: x.split('/')[-1])
summary_data['model_short'] = summary_data['model'].apply(lambda x: x.split('.')[-1].split(':')[0])

# Create faceted plot
g = sns.catplot(
    data=summary_data,
    x='method',
    y='accuracy',
    hue='method',
    col='dataset_short',
    kind='bar',
    palette=palette,
    height=5,
    aspect=1.2,
    alpha=0.85,
    edgecolor='white',
    linewidth=1.5,
    order=methods,
    hue_order=methods,
    legend=False,
    col_wrap=2
)

# Add error bars to each subplot
for i, ax in enumerate(g.axes.flat):
    dataset_short = summary_data['dataset_short'].unique()[i] if i < len(summary_data['dataset_short'].unique()) else None
    if dataset_short is None:
        continue
    
    subset = summary_data[summary_data['dataset_short'] == dataset_short]
    
    for j, method in enumerate(methods):
        method_data = subset[subset['method'] == method]
        if len(method_data) > 0:
            # Average across models/temps for this dataset and method
            avg_acc = method_data['accuracy'].mean()
            avg_lower = method_data['error_lower'].mean()
            avg_upper = method_data['error_upper'].mean()
            
            ax.errorbar(
                j, avg_acc,
                yerr=[[avg_lower], [avg_upper]],
                fmt='none',
                ecolor='black',
                elinewidth=2,
                capsize=4,
                capthick=2,
                alpha=0.7
            )
    
    ax.set_ylim(0, 105)
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_title(f'{dataset_short.upper()}', fontsize=13, fontweight='bold')

g.fig.suptitle(
    f'Accuracy Comparison Across All Datasets\n(with {int(CONFIDENCE_LEVEL*100)}% Bootstrap CIs)', 
    fontsize=16, 
    fontweight='bold',
    y=1.02
)

plt.tight_layout()

# Save summary figure
summary_filename = f"{results_dir}/accuracy_comparison_summary_bootstrap.png"
plt.savefig(summary_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {summary_filename}")
plt.close()


# Create a detailed table with confidence intervals
print("\n" + "=" * 60)
print("CREATING DETAILED RESULTS TABLE")
print("=" * 60)

# Prepare detailed results table
detailed_results = viz_data.copy()
detailed_results['ci_range'] = detailed_results.apply(
    lambda row: f"[{row['lower_ci']:.1f}, {row['upper_ci']:.1f}]", 
    axis=1
)
detailed_results['accuracy_with_ci'] = detailed_results.apply(
    lambda row: f"{row['accuracy']:.1f}% {row['ci_range']}", 
    axis=1
)

# Pivot table for easy viewing
table_data = detailed_results.pivot_table(
    index=['dataset', 'model', 'temperature'],
    columns='method',
    values='accuracy_with_ci',
    aggfunc='first'
)

# Save to CSV
table_filename = f"{results_dir}/detailed_results_with_ci.csv"
table_data.to_csv(table_filename)
print(f"Saved detailed results table: {table_filename}")

# Also save raw data with CI bounds
raw_filename = f"{results_dir}/raw_results_with_ci.csv"
viz_data.to_csv(raw_filename, index=False)
print(f"Saved raw results with CI: {raw_filename}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Number of bootstrap samples: {N_BOOTSTRAP}")
print(f"Confidence level: {CONFIDENCE_LEVEL*100}%")
print(f"All visualizations saved to: {results_dir}")
