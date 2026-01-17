import ast
import concurrent.futures
import hashlib
import os
import pandas as pd
import numpy as np
import time
import requests
from pathlib import Path
from typing import List, Optional, Tuple
import wandb
from tqdm import tqdm

def download_wandb_data_from_multiple_users(
    wandb_usernames: List[str],
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str],
    refresh: bool = False,
    filetype: str = "csv",
    max_workers: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load wandb data from multiple usernames and combine.
    
    Args:
        wandb_usernames: List of wandb usernames to check
        wandb_project_path: Project path (e.g., "spa-experiments")
        data_dir: Directory to store/load data
        sweep_ids: List of sweep IDs to download
        refresh: Whether to re-download data
        filetype: Format to save/load ("csv", "feather", or "parquet")
        max_workers: Number of parallel workers
    
    Returns:
        Tuple of (runs_configs_df, responses_df)
    """
    all_runs_configs = []
    all_responses = []
    
    for username in wandb_usernames:
        print(f"\n{'='*60}")
        print(f"Processing username: {username}")
        print(f"{'='*60}")
        
        try:
            runs_configs = download_wandb_project_runs_configs(
                wandb_project_path=wandb_project_path,
                data_dir=data_dir,
                sweep_ids=sweep_ids,
                refresh=refresh,
                wandb_username=username,
                filetype=filetype,
                max_workers=max_workers,
            )
            all_runs_configs.append(runs_configs)
            print(f"✓ Loaded {len(runs_configs)} run configs from {username}")
        except Exception as e:
            print(f"⚠ Failed to load run configs from {username}: {e}")
        
        try:
            responses = download_wandb_sweep_runs_responses(
                wandb_project_path=wandb_project_path,
                data_dir=data_dir,
                sweep_ids=sweep_ids,
                refresh=refresh,
                wandb_username=username,
                filetype=filetype,
                max_workers=max_workers,
            )
            all_responses.append(responses)
            print(f"✓ Loaded {len(responses)} responses from {username}")
        except Exception as e:
            print(f"⚠ Failed to load responses from {username}: {e}")
    
    # Combine all data
    if all_runs_configs:
        runs_configs_df = pd.concat(all_runs_configs, ignore_index=True)
        # Remove duplicates based on run_id if any
        runs_configs_df = runs_configs_df.drop_duplicates(subset=['run_id'], keep='first')
    else:
        runs_configs_df = pd.DataFrame()
        print("⚠ No run configs loaded from any username")
    
    if all_responses:
        responses_df = pd.concat(all_responses, ignore_index=True)
        # Remove duplicates based on run_id and other key columns if any
        responses_df = responses_df.drop_duplicates(
            subset=['run_id', 'question', 'response_type', 'model_response'], 
            keep='first'
        )
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
    
    return runs_configs_df, responses_df

def setup_notebook_dir(
    notebook_dir: str,
    refresh: bool = False,
) -> Tuple[str, str]:
    """Setup data and results directories."""
    data_dir = os.path.join(notebook_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    results_dir = os.path.join(notebook_dir, "results")
    if refresh:
        import shutil
        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir


def download_wandb_project_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str],
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: Optional[str] = None,
    filetype: str = "csv",
    max_workers: int = 10,
) -> pd.DataFrame:
    """Download run configurations from wandb sweeps."""
    assert filetype in {"csv", "feather", "parquet"}

    api = wandb.Api(timeout=600)

    if wandb_username is None:
        wandb_username = api.viewer.username

    # Include username in filename to avoid collisions
    filename = f"user={wandb_username}_sweeps=" + ",".join(sweep_ids)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    runs_configs_df_path = os.path.join(
        data_dir, hashed_filename + f"_runs_configs.{filetype}"
    )

    if refresh or not os.path.isfile(runs_configs_df_path):
        print(f"Creating {runs_configs_df_path} anew.")

        sweep_results_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_run = {}

            for sweep_id in sweep_ids:
                try:
                    sweep = api.sweep(
                        f"{wandb_username}/{wandb_project_path}/{sweep_id}"
                    )
                    for run in sweep.runs:
                        future_to_run[
                            executor.submit(
                                download_wandb_project_runs_configs_helper, run
                            )
                        ] = run
                except Exception as e:
                    print(f"Error processing sweep {sweep_id}: {str(e)}")

            for future in tqdm(
                concurrent.futures.as_completed(future_to_run), total=len(future_to_run)
            ):
                result = future.result()
                if result is not None:
                    sweep_results_list.append(result)

        runs_configs_df = pd.DataFrame(sweep_results_list)
        runs_configs_df.reset_index(inplace=True, drop=True)

        # Save to disk
        runs_configs_df.to_csv(
            runs_configs_df_path.replace(filetype, "csv"), index=False
        )
        try:
            runs_configs_df.to_feather(
                runs_configs_df_path.replace(filetype, "feather")
            )
        except Exception as e:
            print(f"Error saving to feather: {str(e)}")

        try:
            runs_configs_df.to_parquet(
                runs_configs_df_path.replace(filetype, "parquet"), index=False
            )
        except Exception as e:
            print(f"Error saving to parquet: {str(e)}")

        print(f"Regenerated and wrote {runs_configs_df_path} to disk.")
        del runs_configs_df

    print(f"Reading {runs_configs_df_path} from disk.")
    if filetype == "csv":
        runs_configs_df = pd.read_csv(runs_configs_df_path)
    elif filetype == "feather":
        runs_configs_df = pd.read_feather(runs_configs_df_path)
    elif filetype == "parquet":
        runs_configs_df = pd.read_parquet(runs_configs_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
    print(f"Loaded {runs_configs_df_path} from disk.")

    # Keep only finished runs
    finished_runs = runs_configs_df["State"] == "finished"
    print(
        f"% of successfully finished runs: {100.0 * finished_runs.mean():.2f}% "
        f"({finished_runs.sum()} / {len(finished_runs)})"
    )

    if finished_only:
        runs_configs_df = runs_configs_df[finished_runs]
        assert len(runs_configs_df) > 0, "No finished runs found!"
        runs_configs_df = runs_configs_df.copy()

    return runs_configs_df


def download_wandb_project_runs_configs_helper(run):
    """Helper to download a single run's config."""
    try:
        summary = run.summary._json_dict
        summary.update({k: v for k, v in run.config.items() if not k.startswith("_")})
        summary.update(
            {
                "State": run.state,
                "Sweep": run.sweep.id if run.sweep is not None else None,
                "run_id": run.id,
                "run_name": run.name,
            }
        )
        return summary
    except Exception as e:
        print(f"Error processing run {run.id}: {str(e)}")
        return None


def download_wandb_sweep_runs_responses(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str],
    refresh: bool = False,
    wandb_username: Optional[str] = None,
    filetype: str = "csv",
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Download all individual responses from wandb sweeps.
    Similar to download_wandb_project_runs_histories but for response tables.
    """
    assert filetype in {"csv", "feather", "parquet"}

    api = wandb.Api(timeout=6000)

    if wandb_username is None:
        wandb_username = api.viewer.username

    # Include username in filename to avoid collisions
    filename = f"user={wandb_username}_sweeps=" + ",".join(sweep_ids)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    responses_df_path = os.path.join(
        data_dir, hashed_filename + f"_responses.{filetype}"
    )

    if refresh or not os.path.isfile(responses_df_path):
        print(f"Creating {responses_df_path} anew.")

        all_responses = []
        
        print("Downloading responses from sweeps...")
        for sweep_id in sweep_ids:
            try:
                sweep = api.sweep(f"{wandb_username}/{wandb_project_path}/{sweep_id}")
                print(f"  Processing sweep {sweep_id} with {len(sweep.runs)} runs")
                
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    future_to_run = {
                        executor.submit(
                            download_wandb_sweep_runs_responses_helper,
                            run,
                            sweep_id,
                        ): run
                        for run in sweep.runs
                    }

                    for future in tqdm(
                        concurrent.futures.as_completed(future_to_run),
                        total=len(future_to_run),
                        desc=f"Sweep {sweep_id}"
                    ):
                        run = future_to_run[future]
                        try:
                            responses = future.result()
                            if responses is not None:
                                all_responses.append(responses)
                        except Exception as exc:
                            print(f"{run.id} generated an exception: {exc}")
                            
            except Exception as e:
                print(f"Error processing sweep {sweep_id}: {str(e)}")

        if not all_responses:
            print("No responses found in any runs")
            return pd.DataFrame()

        responses_df = pd.concat(all_responses, ignore_index=True)
        responses_df.reset_index(inplace=True, drop=True)

        # Save to disk
        responses_df.to_csv(
            responses_df_path.replace(filetype, "csv"), index=False
        )
        try:
            responses_df.to_feather(
                responses_df_path.replace(filetype, "feather")
            )
        except Exception as e:
            print(f"Error saving to feather: {str(e)}")

        try:
            responses_df.to_parquet(
                responses_df_path.replace(filetype, "parquet"), index=False
            )
        except Exception as e:
            print(f"Error saving to parquet: {str(e)}")

        print(f"Wrote {responses_df_path} to disk")
        del responses_df

    print(f"Reading {responses_df_path} from disk.")
    if filetype == "csv":
        responses_df = pd.read_csv(responses_df_path)
    elif filetype == "feather":
        responses_df = pd.read_feather(responses_df_path)
    elif filetype == "parquet":
        responses_df = pd.read_parquet(responses_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
    print(f"Loaded {responses_df_path} from disk.")

    return responses_df


def download_wandb_sweep_runs_responses_helper(run, sweep_id):
    """Helper to download responses from a single run."""
    responses = None
    
    try:
        # Get run configuration
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        
        # Try method 1: Get from run history
        for row in run.scan_history(keys=["responses"]):
            if "responses" in row and row["responses"] is not None:
                table_data = row["responses"]
                
                # Check if it's a direct table with data attribute
                if hasattr(table_data, 'data') and hasattr(table_data, 'columns'):
                    df = pd.DataFrame(data=table_data.data, columns=table_data.columns)
                    
                    # Add run metadata
                    df['run_id'] = run.id
                    df['run_name'] = run.name
                    df['run_state'] = run.state
                    df['sweep_id'] = sweep_id
                    
                    # Add all config parameters
                    for k, v in config.items():
                        df[f'config_{k}'] = v
                    
                    return df
        
        # Method 2: Try to get artifacts directly
        for artifact in run.logged_artifacts():
            if artifact.type == "run_table":
                # Download the artifact
                artifact_dir = artifact.download()
                
                # Find the table file
                import json
                for file_path in Path(artifact_dir).rglob("*.json"):
                    with open(file_path, 'r') as f:
                        table_json = json.load(f)
                        if 'data' in table_json and 'columns' in table_json:
                            df = pd.DataFrame(
                                data=table_json['data'],
                                columns=table_json['columns']
                            )
                            
                            # Add run metadata
                            df['run_id'] = run.id
                            df['run_name'] = run.name
                            df['run_state'] = run.state
                            df['sweep_id'] = sweep_id
                            
                            # Add all config parameters
                            for k, v in config.items():
                                df[f'config_{k}'] = v
                            
                            return df
                            
    except Exception as e:
        print(f"Error downloading responses from run {run.id}: {str(e)}")
    
    return None


def calculate_accuracy_metrics(responses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate accuracy metrics from individual responses.
    Aggregates by run and question.
    """
    metrics = []
    
    for (run_id, question_idx), group in tqdm(
        responses_df.groupby(['run_id', 'question_idx']),
        desc="Calculating metrics"
    ):
        # Get direct answers only
        direct_answers = group[group['response_type'] == 'direct_answer']
        
        if len(direct_answers) == 0:
            continue
        
        # Get configuration info
        config_cols = [col for col in group.columns if col.startswith('config_')]
        config_data = group[config_cols].iloc[0].to_dict()
        
        # Get question metadata
        question = group['question'].iloc[0]
        true_answer = group['true_answer'].iloc[0]
        
        # Calculate majority vote
        answer_counts = direct_answers['extracted_answer'].value_counts()
        majority_answer = answer_counts.index[0] if len(answer_counts) > 0 else None
        majority_correct = (majority_answer == true_answer)
        
        # Count answers
        answer_distribution = direct_answers['extracted_answer'].value_counts().to_dict()
        
        metric_row = {
            'run_id': run_id,
            'sweep_id': group['sweep_id'].iloc[0],
            'question_idx': question_idx,
            'question': question,
            'true_answer': true_answer,
            'majority_answer': majority_answer,
            'majority_correct': majority_correct,
            'num_samples': len(direct_answers),
            'answer_distribution': str(answer_distribution),
            **config_data
        }
        
        metrics.append(metric_row)
    
    return pd.DataFrame(metrics)


def main():
    """Main analysis script."""
    # Configuration
    refresh = False  # Set to True to re-download data
    # refresh = True
    
    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir, results_dir = setup_notebook_dir(
        notebook_dir=script_dir,
        refresh=refresh,
    )
    
    # Specify your sweeps
    sweep_ids = [
        "your-sweep-id-here",  # Replace with actual sweep IDs
    ]
    
    # Download run configs
    runs_configs_df = download_wandb_project_runs_configs(
        wandb_project_path="spa-experiments",  # Replace with your project name
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        finished_only=True,
        refresh=refresh,
        wandb_username=wandb.api.default_entity,
    )
    
    print(f"\nLoaded {len(runs_configs_df)} run configurations")
    
    # Download responses from all sweeps
    responses_df = download_wandb_sweep_runs_responses(
        wandb_project_path="spa-experiments",  # Replace with your project name
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=refresh,
        wandb_username=wandb.api.default_entity,
    )
    
    if responses_df.empty:
        print("No data downloaded. Exiting.")
        return
    
    print(f"\nLoaded {len(responses_df)} total responses")
    print(f"Unique runs: {responses_df['run_id'].nunique()}")
    print(f"Unique questions: {responses_df['question_idx'].nunique()}")
    
    # Calculate accuracy metrics
    print("\nCalculating accuracy metrics...")
    metrics_df = calculate_accuracy_metrics(responses_df)
    
    # Save metrics
    metrics_file = os.path.join(data_dir, "accuracy_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Saved metrics to {metrics_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall accuracy by configuration
    if 'config_model_id' in metrics_df.columns:
        for (model, dataset, exp_type), group in metrics_df.groupby([
            'config_model_id', 'config_dataset_name', 'config_experiment_type'
        ]):
            accuracy = group['majority_correct'].mean()
            n_questions = len(group)
            print(f"\nModel: {model}")
            print(f"Dataset: {dataset}")
            print(f"Experiment: {exp_type}")
            print(f"  Majority accuracy: {accuracy:.2%} ({n_questions} questions)")
    
    # Accuracy by number of samples
    if 'config_num_samples' in metrics_df.columns:
        print("\n" + "-"*60)
        print("Accuracy by number of samples:")
        for num_samples, group in metrics_df.groupby('config_num_samples'):
            accuracy = group['majority_correct'].mean()
            print(f"  {num_samples} samples: {accuracy:.2%}")
    
    print("\nFinished running SPA analysis!")
    
    return responses_df, metrics_df, runs_configs_df


if __name__ == "__main__":
    responses_df, metrics_df, runs_configs_df = main()