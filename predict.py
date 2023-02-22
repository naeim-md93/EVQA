import os
import cv2
import torch
from src.preprocess.questions import tokenize_questions, filter_by_tokens_length
from src.utils.pyutils import load_file
from torchvision import transforms as T
from src.models.resnexts import Model, FeatureExtractor


def predict(image_path, question, model, embeddings, idx_to_answer, tokens_length=24, max_ans=5):
    FE = FeatureExtractor()
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    trans = T.Compose([
        T.ToTensor(),
        T.Resize(size=(224, 224)),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img = trans(img).unsqueeze(0)
    img_fea = FE(img)

    q = [{'question': question}]
    q = tokenize_questions(q)
    q = filter_by_tokens_length(q, tokens_length)
    q = q[0]['tokens']

    question = torch.zeros(size=(tokens_length, 300), dtype=torch.float)
    for i in range(len(q)):
        if i < tokens_length:
            if q[i] in embeddings:
                question[i] = embeddings[q[i]]

    question = question.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred = model(img_fea, question)[0]
    pred = torch.softmax(pred, dim=0)

    answers = torch.topk(input=pred, dim=0, k=max_ans)

    for i in range(max_ans):
        print(f'Answer {i + 1}: {idx_to_answer[answers[1][i].item()]}, Probability: {answers[0][i].item()}')


weights_path = 'Model_20.pth'
embeddings = load_file(path=os.path.join(os.getcwd(), 'skipgram_embeddings.pickle'))
answer_to_idx = load_file(path=os.path.join('checkpoints', 'try1', 'preprocess', 'answer_to_idx.pickle'))
idx_to_answer = load_file(path=os.path.join('checkpoints', 'try1', 'preprocess', 'idx_to_answer.pickle'))
tokens_length = 24

model = Model()
weights = torch.load(weights_path)
model.load_state_dict(weights)

# It Should be in test_images path in root
image_path = 'test_images/2.jpeg'
question = 'How many glasses are there'

predict(image_path, question, model, embeddings, idx_to_answer)