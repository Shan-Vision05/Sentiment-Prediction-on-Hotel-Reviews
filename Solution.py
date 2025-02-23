from sklearn.model_selection import train_test_split
import numpy as np
from Util import load_test_data
import torch

pronouns = [
    # 1st Person Pronouns
    "I", "me", "we", "us",  
    "my", "mine", "our", "ours",  
    "myself", "ourselves",  

    # 2nd Person Pronouns
    "you",  
    "your", "yours",  
    "yourself", "yourselves"
]

def split_data(texts, labels):
    length = len(texts)
    return train_test_split(
        texts, 
        labels,
        test_size=0.2,
        random_state= 5
    )

def GetDictionary(path):
    keywords = {}

    with open(path,"r") as f:
        for line in f:
            text = line.strip()
            if text not in keywords.keys():
                keywords[text] = 1
    
    return keywords 



def featurize_text(text):
    
    positive_keywords = GetDictionary('data/positive-words.txt')
    negative_keywords = GetDictionary('data/negative-words.txt')

    x1, x2, x3, x4, x5, x6 = 0, 0, 0, 0, 0, 0

    x6 = len(text.split(' '))

    x6 = np.round(np.log(x6), 4)


    for word in text.split(' '):
        if positive_keywords.get(word) is not None:
            x1 += 1
        if negative_keywords.get(word) is not None:
            x2 += 1
        if word == 'no':
            x3 = 1

        if word in pronouns:
            x4 += 1
    
    for letter in text:
        if letter == '!':
            x5 = 1

    return [x1, x2, x3, x4, x5, x6]

    
    
def normalize(feature_vectors):
    normalized_feature_vestors = []

    for vector in feature_vectors:
        mn, mx = min(vector), max(vector)
        # print(mn, mx)
        norm_vector = [ (i-mn)/(mx-mn) for i in vector]
        normalized_feature_vestors.append(np.round(norm_vector,2))

    return normalized_feature_vestors

####################################################################
####################### Evaluation Code ############################
####################################################################

def Precision(labels, preds):
    
    #calculating True Positives (tp) and False Positives (fp)
    # print(labels)
    tp, fp = 0, 0
    for i in range(len(labels)):
        if preds[i] == 1:
            tp += 1 if labels[i] == 1 else 0
            fp += 1 if labels[i] == 0 else 0

    return np.round(tp/(tp+fp), 4)

def Recall(labels, preds):

    tp, fn = 0, 0
    for i in range(len(labels)):
        if labels[i] == 1:
            tp += 1 if preds[i] == 1 else 0
            fn += 1 if preds[i] == 0 else 0

    return np.round(tp/(tp+fn), 4)

def F1_Score(labels, preds):
    precision = Precision(labels, preds)
    recall = Recall(labels, preds)

    return np.round(2*precision*recall/(precision + recall), 4)


def GetPredictions(model, test_vectors):
    preds = []
    with torch.inference_mode():
        model.eval()
        for i in range(len(test_vectors)):
            log_prob = model(torch.tensor(test_vectors[i], dtype=torch.float32))
            # print(log_prob)
            pred = 1 if log_prob > 0.5 else 0
            preds.append(pred)
    return preds

def Evaluate_Model_onDevSet(model, X_test, y_test):
    
    preds = GetPredictions(model, X_test)

    print(f"\nEvaluation on Dev Test split \n")
    precision = Precision(y_test, preds)
    recall = Recall(y_test, preds)
    f1_score = F1_Score(y_test, preds)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f'F1 Score: {f1_score}')

def Evaluate_Model(model):
    texts, labels = load_test_data('data/HW2-testset.txt')

    vectors = [featurize_text(text) for text in texts]
    vectors = normalize(vectors)

    preds = GetPredictions(model, vectors)

    print(f"\nEvaluation on 'HW2-testset.txt' \n")
    precision = Precision(labels, preds)
    recall = Recall(labels, preds)
    f1_score = F1_Score(labels, preds)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f'F1 Score: {f1_score}')



