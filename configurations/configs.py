import os
from configurations.datasets.VQAv2 import get_VQAv2_configs


def get_configs():

    cfgs = {}

    # ---------------- General Configurations ----------------
    cfgs['session'] = 'try1'
    cfgs['seed'] = 42
    cfgs['root_path'] = os.getcwd()
    cfgs['epochs'] = 50
    cfgs['batch_size'] = 16
    cfgs['lr'] = 1e-5
    # ---------------- Original Dataset Configurations ----------------
    cfgs['datasets'] = {}

    ### VQAv2:
    cfgs['datasets']['VQAv2'] = get_VQAv2_configs(root_path=os.path.join(cfgs['root_path'], 'datasets', 'VQAv2'))
    # ---------------- End of original dataset configurations

    # Checkpoints Configurations
    cfgs['checkpoints_path'] = os.path.join(cfgs['root_path'], 'checkpoints', cfgs['session'])
    cfgs['logs_path'] = os.path.join(cfgs['checkpoints_path'], 'logs')

    # ---------------- Preprocess Configurations ----------------
    cfgs['preprocess'] = {}
    cfgs['preprocess']['root_path'] = os.path.join(cfgs['checkpoints_path'], 'preprocess')
    cfgs['preprocess']['VQAv2_original_data'] = 'VQAv2_original_data.pickle'
    cfgs['preprocess']['skipgram_embeddings'] = 'skipgram_embeddings.pickle'
    cfgs['preprocess']['idx_to_answer'] = 'idx_to_answer.pickle'
    cfgs['preprocess']['answer_to_idx'] = 'answer_to_idx.pickle'
    cfgs['preprocess']['image_features'] = os.path.join(cfgs['preprocess']['root_path'], 'image_features')
    # ---------------- End of Preprocess Configurations ----------------

    # Create Necessary Directories
    os.makedirs(name=cfgs['checkpoints_path'], exist_ok=True)
    os.makedirs(name=cfgs['logs_path'], exist_ok=True)
    os.makedirs(name=cfgs['preprocess']['root_path'], exist_ok=True)
    os.makedirs(name=cfgs['preprocess']['image_features'], exist_ok=True)

    return cfgs



