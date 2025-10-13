import boto3
import pandas as pd
import re
import asyncio
import random
import wandb
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from tqdm.asyncio import tqdm
from botocore.exceptions import ClientError

# ============================================================================
# wandb Integration
# ============================================================================
def init_wandb():
    """Initialize wandb with sweep configuration or defaults."""
    wandb.init()
    config = wandb.config
    return config

# ============================================================================
# Setup AWS Bedrock Client
# ============================================================================
session = boto3.session.Session()
client = session.client('bedrock-runtime', region_name='us-west-2')

def load_and_prepare_dataset(dataset_name, max_questions):
    """Load and filter dataset based on configuration."""
    print(f"Loading dataset: {dataset_name}")
    
    if dataset_name == 'cais/hle':
        dataset = load_dataset(dataset_name)
        df = dataset['test'].to_pandas()
        df = df[(df['answer_type'] == 'exactMatch') & (df['answer'].str.lower().isin(['yes', 'no']))]
        answer_type = 'yes_no'
        answer_options = ['yes', 'no']
        print("Using HLE dataset with YES/NO questions")
        
    elif dataset_name == 'google/boolq':
        dataset = load_dataset(dataset_name)
        df = dataset['validation'].to_pandas()
        df['answer'] = df['answer'].apply(lambda x: 'true' if x else 'false')
        answer_type = 'true_false'
        answer_options = ['true', 'false']
        print("Using BoolQ dataset with TRUE/FALSE questions")
        
    elif dataset_name == 'tasksource/strategy-qa':
        # Load from JSON files - only has test split
        dataset = load_dataset('json', data_files={
            'test': 'hf://datasets/tasksource/strategy-qa/test.jsonl'
        })
        df = dataset['test'].to_pandas()
        # Ensure answer column is lowercase string 'true' or 'false'
        df['answer'] = df['answer'].astype(str).str.lower()
        answer_type = 'true_false'
        answer_options = ['true', 'false']
        print("Using Strategy-QA dataset with TRUE/FALSE questions")
        
    elif dataset_name == 'tasksource/com2sense':
        dataset = load_dataset(dataset_name)
        df = dataset['validation'].to_pandas()
        # Rename 'sent' to 'question' and 'label' to 'answer'
        df = df.rename(columns={'sent': 'question', 'label': 'answer'})
        # Convert label (0/1) to 'true'/'false' strings
        df['answer'] = df['answer'].apply(lambda x: 'true' if x == 1 else 'false')
        answer_type = 'true_false'
        answer_options = ['true', 'false']
        print("Using Com2Sense dataset with TRUE/FALSE questions (does scenario make sense?)")
        
    elif dataset_name == 'kyssen/predict-the-futurebench-cutoff-June25':
        dataset = load_dataset(dataset_name)
        df = dataset['train'].to_pandas()
        # Normalize answer to lowercase yes/no
        df['answer'] = df['answer'].str.strip().str.lower()
        answer_type = 'yes_no'
        answer_options = ['yes', 'no']
        print("Using Predict-the-Future Bench dataset with YES/NO questions")
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if max_questions is not None:
        df = df.head(max_questions)
        print(f"Limited to first {max_questions} questions")
    
    print(f"Total questions to process: {len(df)}")
    return df, answer_type, answer_options

async def call_llm(prompt, model_id, config, executor, semaphore):
    """Call LLM with exponential backoff retry logic."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        msgs = [{'role': 'user', 'content': [{'text': prompt}]}]
        
        for attempt in range(config.max_retries):
            try:
                if config.verbose_logging:
                    print(f"  → API call to {model_id.split('.')[-1]} (attempt {attempt+1})")
                
                res = await loop.run_in_executor(executor, lambda: client.converse(
                    modelId=model_id, 
                    messages=msgs, 
                    system=[], 
                    inferenceConfig={
                        'temperature': config.temperature,
                        'maxTokens': config.max_tokens
                    }
                ))
                
                content = res['output']['message']['content']
                thinking, text = "", ""
                for item in content:
                    if isinstance(item, dict):
                        if 'reasoningContent' in item:
                            thinking = item['reasoningContent'].get('reasoningText', {}).get('text', '')
                        elif 'text' in item:
                            text = item['text']
                
                if config.verbose_logging:
                    print(f"  ✓ Response received")
                
                return text, thinking
            
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code in ['ThrottlingException', 'TooManyRequestsException']:
                    if attempt < config.max_retries - 1:
                        wait_time = (config.backoff_base ** attempt) + random.uniform(0, config.backoff_jitter_max)
                        print(f"  ⚠ Rate limited (attempt {attempt+1}/{config.max_retries}). Waiting {wait_time:.2f}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"  ✗ Max retries exceeded due to rate limiting")
                        raise
                else:
                    raise
            
            except Exception as e:
                print(f"  ✗ Unexpected error in call_llm: {e}")
                raise
        
        raise Exception(f"Max retries exceeded for model {model_id}")

def create_prompts(question, answer_type, experiment_type):
    """Create prompts based on experiment type."""
    if answer_type == 'yes_no':
        direct_prompt = f"{question}\nAnswer with just YES or NO:"
        
        if experiment_type == 'surprisingly_popular':
            prediction_prompt = f'''Question: {question}
What percentage of language models will answer YES to this question? Give just a number 0-100.
What percentage will answer NO? (should sum to 100)
Format: YES:XX NO:XX'''
        else:  # confidence_weighted
            prediction_prompt = f'''Question: {question}
Answer YES or NO, then rate your confidence from 0-100.
Format: Answer:YES Confidence:XX or Answer:NO Confidence:XX'''
    
    else:  # true_false
        direct_prompt = f"{question}\nAnswer with just TRUE or FALSE:"
        
        if experiment_type == 'surprisingly_popular':
            prediction_prompt = f'''Question: {question}
What percentage of language models will answer TRUE to this question? Give just a number 0-100.
What percentage will answer FALSE? (should sum to 100)
Format: TRUE:XX FALSE:XX'''
        else:  # confidence_weighted
            prediction_prompt = f'''Question: {question}
Answer TRUE or FALSE, then rate your confidence from 0-100.
Format: Answer:TRUE Confidence:XX or Answer:FALSE Confidence:XX'''
    
    return direct_prompt, prediction_prompt

def parse_spa_response(pred_text, answer_type, answer_options):
    """Parse surprisingly popular answer response."""
    if answer_type == 'yes_no':
        option1_match = re.search(r'YES:?\s*(\d+)', pred_text, re.I)
        option2_match = re.search(r'NO:?\s*(\d+)', pred_text, re.I)
    else:  # true_false
        option1_match = re.search(r'TRUE:?\s*(\d+)', pred_text, re.I)
        option2_match = re.search(r'FALSE:?\s*(\d+)', pred_text, re.I)
    
    pred_option1 = int(option1_match.group(1))/100 if option1_match else 0.5
    pred_option2 = int(option2_match.group(1))/100 if option2_match else 0.5
    
    return pred_option1, pred_option2

def parse_confidence_response(pred_text, answer_type, answer_options):
    """Parse confidence-weighted response."""
    answer_match = re.search(r'Answer:?\s*(yes|no|true|false)', pred_text, re.I)
    confidence_match = re.search(r'Confidence:?\s*(\d+)', pred_text, re.I)
    
    answer = answer_match.group(1).lower() if answer_match else None
    confidence = int(confidence_match.group(1))/100 if confidence_match else 0.5
    
    return answer, confidence

async def process_question(row, q_idx, config, executor, semaphore, answer_type, answer_options):
    """Process a single question with the configured number of samples."""
    try:
        q, true_ans = row['question'], row['answer'].lower()
        
        if config.verbose_logging:
            print(f"\n[Q{q_idx}] Processing: {q[:80]}...")
        
        # Special handling for com2sense - ask if scenario makes sense
        if config.dataset_name == 'tasksource/com2sense':
            direct_prompt = f"{q}\nDoes this scenario make sense? Answer with just TRUE (makes sense) or FALSE (doesn't make sense):"
            
            if config.experiment_type == 'surprisingly_popular':
                prediction_prompt = f'''Scenario: {q}
What percentage of language models will answer TRUE (makes sense) to this scenario? Give just a number 0-100.
What percentage will answer FALSE (doesn't make sense)? (should sum to 100)
Format: TRUE:XX FALSE:XX'''
            else:  # confidence_weighted
                prediction_prompt = f'''Scenario: {q}
Does this scenario make sense? Answer TRUE (makes sense) or FALSE (doesn't make sense), then rate your confidence from 0-100.
Format: Answer:TRUE Confidence:XX or Answer:FALSE Confidence:XX'''
        
        # Special handling for predict-the-future - add uncertainty caveat
        elif config.dataset_name == 'kyssen/predict-the-futurebench-cutoff-June25':
            direct_prompt = f"{q}\nYou may not know the answer for certain, but please make your best prediction. Answer with just YES or NO:"
            
            if config.experiment_type == 'surprisingly_popular':
                prediction_prompt = f'''Question: {q}
What percentage of language models will answer YES to this question? Give just a number 0-100.
What percentage will answer NO? (should sum to 100)
Note: The answer may not be knowable for certain, but predict based on available information.
Format: YES:XX NO:XX'''
            else:  # confidence_weighted
                prediction_prompt = f'''Question: {q}
You may not know the answer for certain, but please make your best prediction. Answer YES or NO, then rate your confidence from 0-100.
Format: Answer:YES Confidence:XX or Answer:NO Confidence:XX'''
        
        else:
            direct_prompt, prediction_prompt = create_prompts(q, answer_type, config.experiment_type)
        
        # Direct answer queries
        if config.verbose_logging:
            print(f"[Q{q_idx}] Starting {config.num_samples} direct answer calls...")
        
        prompts = [direct_prompt for _ in range(config.num_samples)]
        results = await asyncio.gather(*[call_llm(p, config.model_id, config, executor, semaphore) for p in prompts])
        
        # Prediction/Confidence queries
        if config.verbose_logging:
            print(f"[Q{q_idx}] Starting {config.num_samples} prediction calls...")
        
        pred_results = await asyncio.gather(*[call_llm(prediction_prompt, config.model_id, config, executor, semaphore) for _ in range(config.num_samples)])
        
        # Process direct responses
        direct_responses = []
        option1_count, option2_count = 0, 0
        
        for i, (response, thinking) in enumerate(results):
            response_lower = response.lower()
            if answer_type == 'yes_no':
                extracted = 'yes' if 'yes' in response_lower else 'no' if 'no' in response_lower else 'unclear'
            else:
                extracted = 'true' if 'true' in response_lower else 'false' if 'false' in response_lower else 'unclear'
            
            if extracted == answer_options[0]: option1_count += 1
            elif extracted == answer_options[1]: option2_count += 1
            
            direct_responses.append({
                'model': config.model_id,
                'question_idx': q_idx,
                'response_type': 'direct_answer',
                'response_idx': i,
                'question': q,
                'true_answer': true_ans,
                'thinking': thinking,
                'model_response': response,
                'extracted_answer': extracted
            })
        
        # Process prediction responses based on experiment type
        if config.experiment_type == 'surprisingly_popular':
            pred_option1_values, pred_option2_values = [], []
            
            for i, (pred_text, pred_thinking) in enumerate(pred_results):
                pred_option1, pred_option2 = parse_spa_response(pred_text, answer_type, answer_options)
                pred_option1_values.append(pred_option1)
                pred_option2_values.append(pred_option2)
                
                direct_responses.append({
                    'model': config.model_id,
                    'question_idx': q_idx,
                    'response_type': 'prediction',
                    'response_idx': i,
                    'question': q,
                    'true_answer': true_ans,
                    'thinking': pred_thinking,
                    'model_response': pred_text,
                    'extracted_answer': f"{answer_options[0].upper()}:{pred_option1:.2f} {answer_options[1].upper()}:{pred_option2:.2f}"
                })
            
            # Calculate SPA metrics
            avg_pred_option1 = sum(pred_option1_values) / len(pred_option1_values)
            actual_option1 = option1_count / (option1_count + option2_count) if (option1_count + option2_count) > 0 else 0.5
            
            option1_surprise = avg_pred_option1 - actual_option1
            option2_surprise = (1 - avg_pred_option1) - (1 - actual_option1)
            
            spa_ans = answer_options[0] if option1_surprise < option2_surprise else answer_options[1]
            spa_correct = (spa_ans == true_ans)
            
            method_answer = spa_ans
            method_correct = spa_correct
            method_specific = {
                f'avg_predicted_{answer_options[0]}_rate': avg_pred_option1,
                f'actual_{answer_options[0]}_rate': actual_option1,
                'spa_answer': spa_ans,
                'spa_correct': spa_correct
            }
        
        else:  # confidence_weighted
            confidence_weights = {answer_options[0]: 0.0, answer_options[1]: 0.0}
            
            for i, (pred_text, pred_thinking) in enumerate(pred_results):
                answer, confidence = parse_confidence_response(pred_text, answer_type, answer_options)
                
                if answer in answer_options:
                    confidence_weights[answer] += confidence
                
                direct_responses.append({
                    'model': config.model_id,
                    'question_idx': q_idx,
                    'response_type': 'confidence',
                    'response_idx': i,
                    'question': q,
                    'true_answer': true_ans,
                    'thinking': pred_thinking,
                    'model_response': pred_text,
                    'extracted_answer': f"Answer:{answer} Confidence:{confidence:.2f}" if answer else "unclear"
                })
            
            # Confidence-weighted answer
            cw_ans = max(confidence_weights, key=confidence_weights.get)
            cw_correct = (cw_ans == true_ans)
            
            method_answer = cw_ans
            method_correct = cw_correct
            method_specific = {
                f'{answer_options[0]}_confidence_weight': confidence_weights[answer_options[0]],
                f'{answer_options[1]}_confidence_weight': confidence_weights[answer_options[1]],
                'cw_answer': cw_ans,
                'cw_correct': cw_correct
            }
        
        # Common metrics
        maj_ans = answer_options[0] if option1_count > option2_count else answer_options[1]
        maj_correct = (maj_ans == true_ans)
        
        if config.verbose_logging:
            print(f"[Q{q_idx}] ✓ Complete - Majority:{maj_ans} ({'✓' if maj_correct else '✗'}), Method:{method_answer} ({'✓' if method_correct else '✗'})")
        
        summary = {
            'model': config.model_id,
            'experiment_type': config.experiment_type,
            'question_idx': q_idx,
            'question': q,
            'true_answer': true_ans,
            f'{answer_options[0]}_count': option1_count,
            f'{answer_options[1]}_count': option2_count,
            'majority_answer': maj_ans,
            'majority_correct': maj_correct,
            'method_answer': method_answer,
            'method_correct': method_correct,
            **method_specific,
            'error': None
        }
        
        return summary, direct_responses
        
    except Exception as e:
        print(f"[Q{q_idx}] ✗ Error: {e}")
        return {'model': config.model_id, 'question_idx': q_idx, 'error': str(e)}, []

async def run_experiment(config):
    """Main experiment loop."""
    # Load dataset
    df, answer_type, answer_options = load_and_prepare_dataset(
        config.dataset_name, 
        config.max_questions
    )
    
    # Setup rate limiting
    executor = ThreadPoolExecutor(max_workers=config.thread_pool_workers)
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    print(f"\n{'='*60}\nProcessing model: {config.model_id}\n{'='*60}")
    
    try:
        tasks = [
            process_question(row, idx, config, executor, semaphore, answer_type, answer_options)
            for idx, (_, row) in enumerate(df.iterrows())
        ]
        results = await tqdm.gather(*tasks, desc=f"{config.model_id}")
        
        summaries = [r[0] for r in results]
        all_responses = [resp for r in results for resp in r[1]]
        
        # Filter successful summaries
        successful_summaries = [s for s in summaries if s.get('error') is None]
        
        if successful_summaries:
            df_sum = pd.DataFrame(successful_summaries)
            df_resp = pd.DataFrame(all_responses)
            
            # Log individual responses as wandb Table
            wandb.log({
                "responses": wandb.Table(dataframe=df_resp)
            })
            
            # Calculate and log basic metrics for monitoring
            metrics = {
                'num_questions_processed': len(df_sum),
                'num_questions_failed': len(summaries) - len(successful_summaries),
                'majority_accuracy': df_sum['majority_correct'].mean(),
                'method_accuracy': df_sum['method_correct'].mean(),
            }
            
            if config.experiment_type == 'surprisingly_popular':
                metrics['spa_accuracy'] = df_sum['spa_correct'].mean()
            else:
                metrics['cw_accuracy'] = df_sum['cw_correct'].mean()
            
            wandb.log(metrics)
            
            print(f"\nResults: {len(df_sum)} questions processed")
            print(f"  Majority accuracy: {metrics['majority_accuracy']:.2%}")
            print(f"  Method accuracy: {metrics['method_accuracy']:.2%}")
        
        executor.shutdown(wait=True)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        wandb.log({"error": str(e)})
        raise

def main():
    """Entry point for wandb sweep."""
    config = init_wandb()
    asyncio.run(run_experiment(config))
    wandb.finish()

if __name__ == "__main__":
    main()