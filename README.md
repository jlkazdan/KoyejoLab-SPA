Instructions for running the code:
1.  `conda create -n spa-2025-10`
2.  `conda activate spa-2025-10`
3.  `cd Koyejolab-Polling-Techniques-For-LMs`
4.  `pip install -r requirements.txt`
5.  `export PYTHONPATH=.`
6.  Create a new wandb sweep.  
`wandb sweep sweeps/spa_binary_kyssen_dataset_vllm.yaml`
You will need to replace the entity with your own entity.  This requires a wandb account.  
7.  Run the agent that the sweep command creates.  
