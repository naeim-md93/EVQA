import os
import gc
import numpy as np

import torch
from torch.backends import cudnn
from torchvision import transforms as T

from src.val import val
from src.train import train
from src.models.resnexts import Model
from configurations.configs import get_configs
from src.datasets.VQAv2 import get_data_loader
from src.utils.imgutils import extract_features
from src.utils.pyutils import set_seed, save_file
from src.utils.datautils import get_original_data
from src.preprocess.data import remove_repeated_data
from src.preprocess.questions import (
    tokenize_questions,
    get_tokens_length_frequency,
    filter_by_tokens_length,
    get_token_embeddings,
)
from src.preprocess.annotations import (
    get_answers_frequency,
    get_k_frequent_answers,
    filter_by_k_frequent_answers,
    get_answers_dictionary
)


# Enable CuDNN
cudnn.enabled = True
cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get Configurations
cfgs = get_configs()

# Random State
set_seed(seed=cfgs['seed'])

# Getting Original Data
original_data = get_original_data(cfgs=cfgs)
print('Getting Original Data:')
print(f'Train Data: {len(original_data["VQAv2"]["train"])}')
print(f'Val Data: {len(original_data["VQAv2"]["val"])}')

# Split Train and Val Data
train_data = original_data["VQAv2"]["train"]
val_data = original_data["VQAv2"]["val"]
print('Analyzing Data ...')
print(f'Train Data: {len(train_data)}')
print(f'# Unique question_id: {len(set([d["question_id"] for d in train_data]))}')
print(f'# Unique image_id: {len(set([d["image_id"] for d in train_data]))}')
print(f'# Unique answer: {len(set([d["answer"] for d in train_data]))}')
print(f'# Unique question: {len(set([d["question"] for d in train_data]))}')
print(f'# Unique (A, Q): {len(set([(d["answer"], d["question"]) for d in train_data]))}')
print(f'# Unique (A, Q, I): {len(set([(d["answer"], d["question"], d["image_id"]) for d in train_data]))}')


# Preprocessing Data:
## Removing repeated data:
train_data = remove_repeated_data(data=train_data)
print('After removing repeated data ...')
print(f'Train Data: {len(train_data)}')
print(f'# Unique question_id: {len(set([d["question_id"] for d in train_data]))}')
print(f'# Unique image_id: {len(set([d["image_id"] for d in train_data]))}')
print(f'# Unique answer: {len(set([d["answer"] for d in train_data]))}')
print(f'# Unique question: {len(set([d["question"] for d in train_data]))}')
print(f'# Unique (A, Q): {len(set([(d["answer"], d["question"]) for d in train_data]))}')
print(f'# Unique (A, Q, I): {len(set([(d["answer"], d["question"], d["image_id"]) for d in train_data]))}')

## Filter data by k frequent answers
dataset = train_data + val_data
answers_frequency = get_answers_frequency(data=dataset)
k_frequent_answers = get_k_frequent_answers(answers_frequency=answers_frequency, k=1000)
train_data = filter_by_k_frequent_answers(data=train_data, answers=k_frequent_answers)
idx_to_answer, answer_to_idx = get_answers_dictionary(
    answers=k_frequent_answers,
    save_path=cfgs['preprocess']['root_path'],
    idx2ans_save_name=cfgs['preprocess']['idx_to_answer'],
    ans2idx_save_name=cfgs['preprocess']['answer_to_idx'],
)

print(f'Dataset: {len(dataset)}')
print(f'Number of unique answers: {len(answers_frequency)}')
print(f'{len(k_frequent_answers)}/{len(answers_frequency)} = {len(k_frequent_answers) / len(answers_frequency)}')
print(f'{sum([v for k, v in k_frequent_answers.items()])}/{len(dataset)} = {(sum([v for k, v in k_frequent_answers.items()]))/len(dataset)}')
print(f'Max: {max(k_frequent_answers.items(), key=lambda x: x[1])}')
print(f'Min: {min(k_frequent_answers.items(), key=lambda x: x[1])}')
print('After filtering by k frequent answers ...')
print(f'Train Data: {len(train_data)}')
print(f'# Unique question_id: {len(set([d["question_id"] for d in train_data]))}')
print(f'# Unique image_id: {len(set([d["image_id"] for d in train_data]))}')
print(f'# Unique answer: {len(set([d["answer"] for d in train_data]))}')
print(f'# Unique question: {len(set([d["question"] for d in train_data]))}')
print(f'# Unique (A, Q): {len(set([(d["answer"], d["question"]) for d in train_data]))}')
print(f'# Unique (A, Q, I): {len(set([(d["answer"], d["question"], d["image_id"]) for d in train_data]))}')


# Preprocessing Train Data
## Step 1: Preprocessing Questions
### Tokenize Questions
train_data = tokenize_questions(data=train_data)
tokens_length_frequency = get_tokens_length_frequency(data=train_data)
print(f'Tokens Length Freqs: {tokens_length_frequency}')
print(f'Train Data: {len(train_data)}')
print(f'# Unique question_id: {len(set([d["question_id"] for d in train_data]))}')
print(f'# Unique image_id: {len(set([d["image_id"] for d in train_data]))}')
print(f'# Unique answer: {len(set([d["answer"] for d in train_data]))}')
print(f'# Unique question: {len(set([d["question"] for d in train_data]))}')
print(f'# Unique tokens: {len(set([t for d in train_data for t in d["tokens"]]))}')
print(f'# Unique (A, T): {len(set([(d["answer"], tuple(d["tokens"])) for d in train_data]))}')
print(f'# Unique (A, T, I): {len(set([(d["answer"], tuple(d["tokens"]), d["image_id"]) for d in train_data]))}')

### Limit Questions by length
tokens_length = 24
train_data = filter_by_tokens_length(data=train_data, tokens_length=tokens_length)
print(f'Train Data: {len(train_data)}')
print(f'# Unique question_id: {len(set([d["question_id"] for d in train_data]))}')
print(f'# Unique image_id: {len(set([d["image_id"] for d in train_data]))}')
print(f'# Unique answer: {len(set([d["answer"] for d in train_data]))}')
print(f'# Unique question: {len(set([d["question"] for d in train_data]))}')
print(f'# Unique tokens: {len(set([t for d in train_data for t in d["tokens"]]))}')
print(f'# Unique (A, T): {len(set([(d["answer"], tuple(d["tokens"])) for d in train_data]))}')
print(f'# Unique (A, T, I): {len(set([(d["answer"], tuple(d["tokens"]), d["image_id"]) for d in train_data]))}')

# Preprocessing Validation Data:
val_data = filter_by_k_frequent_answers(data=val_data, answers=k_frequent_answers)
val_data = tokenize_questions(data=val_data)
val_data = filter_by_tokens_length(data=val_data, tokens_length=tokens_length)


token_embeddings = get_token_embeddings(
    save_path=cfgs['root_path'],
    save_name=cfgs['preprocess']['skipgram_embeddings'],
)

# Extract Features:
train_trans = T.Compose([
    T.ToTensor(),
    T.Resize(size=(224, 224)),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

val_trans = T.Compose([
    T.ToTensor(),
    T.Resize(size=(224, 224)),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_data = extract_features(
    data=train_data,
    trans=train_trans,
    device=device,
    save_path=cfgs['preprocess']['image_features']
)

val_data = extract_features(
    data=val_data,
    trans=val_trans,
    device=device,
    save_path=cfgs['preprocess']['image_features']
)

# Preparing Datasets and Dataloaders:
trainloader = get_data_loader(
    data=train_data,
    embeddings=token_embeddings,
    answer_to_idx=answer_to_idx,
    tokens_length=tokens_length,
    trans=train_trans,
    batch_size=cfgs['batch_size'],
    shuffle=True
)
valloader = get_data_loader(
    data=val_data,
    embeddings=token_embeddings,
    answer_to_idx=answer_to_idx,
    tokens_length=tokens_length,
    trans=val_trans,
    batch_size=cfgs['batch_size'],
    shuffle=False
)

gc.collect()
torch.cuda.empty_cache()

model = Model()

best_loss = np.inf
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=cfgs['lr'])
model = model.to(device)

history = {
    'train_accuracy': [],
    'train_loss': [],
    'val_accuracy': [],
    'val_loss': []
}

for i in range(cfgs['epochs']):
    print(f'Epoch: {i + 1}')

    model, train_scores = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloader=trainloader,
        device=device,
    )
    history['train_accuracy'].append(train_scores['accuracy'])
    history['train_loss'].append(train_scores['loss'])

    val_scores = val(
        model=model,
        criterion=criterion,
        dataloader=valloader,
        device=device,
    )
    history['val_accuracy'].append(val_scores['accuracy'])
    history['val_loss'].append(val_scores['loss'])

    if train_scores['loss'] < best_loss:
        best_loss = train_scores['loss']
        save_file(data=history, path=cfgs['root_path'], file_name=f'History_{i + 1}.pickle')
        torch.save(obj=model.state_dict(), f=os.path.join(cfgs['root_path'], f'Model_{i + 1}.pth'))


