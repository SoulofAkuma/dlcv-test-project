import os
import json
import time
import torch
from training import train

CONFIG_PATH = '/scratch/vihps/vihps01/configs'
RESULTS_PATH = '/scratch/vihps/vihps01/results'

job_index = os.environ['SLURM_PROCID']
config = None
with open(os.path.join(CONFIG_PATH, f'config_{job_index}.json'), 'r') as file:
    config = json.load(file)

device_count = torch.cuda.device_count()
device = torch.device(f'cuda:{int(job_index) % device_count}'
                      if torch.cuda.is_available() else 'cpu')

train_start = time.time()
accuracies, durations = train(device, config['batch_size'], config['patch_size'],
                              config['embedding_dim'], config['learning_rate'],
                              config['epochs'], config['dataset_root'], False)
results = {
    'config': config,
    'accuracies_by_epoch': accuracies,
    'durations_by_epoch': durations,
    'training_duration': time.time() - train_start()
}

os.makedirs(RESULTS_PATH, exist_ok=True)
with open(os.path.join(RESULTS_PATH, f'results_{job_index}.json'), 'w+') as file:
    json.dump(results, file)