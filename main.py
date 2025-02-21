from Solution import *
from Trainer import Train
from Util import *

from LogisticRegressionModel import *

def Load_and_Vectorize_Data():

    all_texts, all_labels = load_train_data('data/hotelPosT-train.txt', 'data/hotelNegT-train.txt')
    
    ##### split_data is defined in Solution.py #####
    train_texts,  dev_texts, train_labels, dev_labels = split_data(all_texts, all_labels)

    ## Vectorization of inputs

    # Featurize and normalize: Both functions defined in Solution.py #
    train_vectors = [featurize_text(text) for text in train_texts]
    train_vectors = normalize(train_vectors)

    # Featurize and normalize
    test_vectors = [featurize_text(text) for text in dev_texts]
    test_vectors = normalize(test_vectors)

    return train_vectors, train_labels, test_vectors, dev_labels

def main():
    
    model = SentimentClassifier() # Using Default Dimensions

    X_train, y_train, X_test, y_test = Load_and_Vectorize_Data()

    Train(model, X_train, y_train, X_test, y_test)

    ## this function is responsible for calculating Precision, Recall and F1-Score
    Evaluate_Model(model)

if __name__ == '__main__':
    main()
    




    



    