import os


def get_VQAv2_configs(root_path):

    cfgs = {}

    # ---------------- Original Dataset Configurations ----------------

    # Train Data Configuration
    cfgs['train'] = {}
    cfgs['train']['annotations'] = {
        'file_name': 'v2_mscoco_train2014_annotations.json',
        'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',
        'path': os.path.join(root_path, 'train')
    }

    cfgs['train']['images'] = {
        'file_name': 'train2014',
        'link': 'http://images.cocodataset.org/zips/train2014.zip',
        'path': os.path.join(root_path, 'train')
    }

    cfgs['train']['open_ended_questions'] = {
        'file_name': 'v2_OpenEnded_mscoco_train2014_questions.json',
        'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',
        'path': os.path.join(root_path, 'train')
    }

    # Val Data Configuration
    cfgs['val'] = {}
    cfgs['val']['annotations'] = {
        'file_name': 'v2_mscoco_val2014_annotations.json',
        'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
        'path': os.path.join(root_path, 'val')
    }

    cfgs['val']['images'] = {
        'file_name': 'val2014',
        'link': 'http://images.cocodataset.org/zips/val2014.zip',
        'path': os.path.join(root_path, 'val')
    }
    
    cfgs['val']['open_ended_questions'] = {
        'file_name': 'v2_OpenEnded_mscoco_val2014_questions.json',
        'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
        'path': os.path.join(root_path, 'val')
    }

    os.makedirs(name=cfgs['train']['annotations']['path'], exist_ok=True)
    os.makedirs(name=cfgs['train']['images']['path'], exist_ok=True)
    os.makedirs(name=cfgs['train']['open_ended_questions']['path'], exist_ok=True)
    os.makedirs(name=cfgs['val']['annotations']['path'], exist_ok=True)
    os.makedirs(name=cfgs['val']['images']['path'], exist_ok=True)
    os.makedirs(name=cfgs['val']['open_ended_questions']['path'], exist_ok=True)

    return cfgs
