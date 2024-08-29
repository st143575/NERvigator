import numpy as np
import pandas as pd
from collections import defaultdict, Counter

class NaiveBayesClassifier:

    def __init__(self, classes):
        self.classes = classes
        self.priors = np.ones(len(classes))
        self.likelihoods = {cls: 1 for cls in self.classes}
        self.vocabulary = set()

    def fit(self, X_train, y_train):
        
        # Calculate class priors
        counts = np.bincount(y_train, minlength=len(self.classes))
        self.priors *= counts / len(y_train)

        cls_feature_counts = {cls: Counter() for cls in self.classes}
        for sample, label in zip(X_train, y_train):
            self.vocabulary.update(sample)
            cls_feature_counts[label].update(sample)

        # Compute likelihoods with Laplace smoothing
        for cls in self.classes:
            total_count = sum(cls_feature_counts[cls].values()) + len(self.vocabulary)
            self.likelihoods[cls] = {token: (count + 1) / total_count for token, count in cls_feature_counts[cls].items()}


    def predict(self, X):
            
        predictions = []
        for sample in X:

            posteriors = []
            for cls in self.classes:
                
                prior = np.log(self.priors[cls])
                likelihood = np.sum([np.log(self.likelihoods[cls].get(sample, 1 / (sum(self.likelihoods[cls].values()) + len(self.vocabulary)))) for sample in X])
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            predictions.append(self.classes[np.argmax(posteriors)])
        
        return predictions
