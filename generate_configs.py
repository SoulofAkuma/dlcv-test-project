import os
from argparse import ArgumentParser
import json

DATASET_PATH = '/scratch/vihps/vihps01/data'
PATCH_SIZES = [12, 7, 4, 2]
EMBEDDING_DIMENSIONS = [384, 192]
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

def create_configs(dir: str):

    os.makedirs(dir, exist_ok=True)

    for i in range(len(PATCH_SIZES)*len(EMBEDDING_DIMENSIONS)):
        config = {}
        patch_index = i % len(PATCH_SIZES)
        embedding_index = i // len(PATCH_SIZES)
        
        config['epochs'] = EPOCHS
        config['dataset_root'] = DATASET_PATH
        config['batch_size'] = BATCH_SIZE
        config['learning_rate'] = LEARNING_RATE
        config['patch_size'] = PATCH_SIZES[patch_index]
        config['embedding_dim'] = EMBEDDING_DIMENSIONS[embedding_index]

        with open(os.path.join(dir, f'config_{i}.json'), 'w+') as file:
            json.dump(config, file)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()
    create_configs(os.path.abspath(args.output_dir))
