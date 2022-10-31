# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:40:22 2022

@author: andre
"""
import pandas as pd
import numpy as np
df = pd.read_csv('data/lingspam-emails.csv.bz2', sep='\t')
df = df.dropna(subset=['spam', 'message'])

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(binary=True)
# define vectorizer

X = vectorizer.fit_transform(df.message).toarray()
y = df['spam'].to_numpy()*1
# vectorize your data. Note: this creates a sparse matrix,
# use .toarray() if you run into trouble

vocabulary = vectorizer.get_feature_names()
# in case you want to see what are the actual words

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.matrix(X), y, test_size=0.2, random_state=25)

# Variable Definition:
# Pr_S1 : Pr(category = S)  
# Pr_S0 : Pr(category = NS)  
# Pr_W1 : Pr(w = 1)  
# Pr_W0 : Pr(w = 0)  
# Pr_W1_S1 : Pr(w = 1 | category = S)  
# Pr_W1_S0 : Pr(w = 1 | category = NS)  
# Pr_W0_S1 : Pr(w = 0 | category = S)  
# Pr_W0_S0 : Pr(w = 0 | category = NS)  

# ------------------------------------
# ln_Pr_W1 : logPr(W = 1)  
# lnL_S1_W : l(category = S | W)

numSpam = len(df[df['spam'] == True]) 
numNonSpam = len(df[df['spam'] == False])
total = len(df)

naive_model_accuracy = max(numSpam, numNonSpam)/total
print("A naive model would predict towards the majority, with an accuracy of " + str(naive_model_accuracy) + " in this case.")

numSpam = len(X_train[y_train == 1]) 
numNonSpam = len(X_train[y_train == 0])
total = len(X_train)

Pr_S1 = numSpam / total
Pr_S0 = numNonSpam / total

# Using natural log likelihood:
ln_Pr_S1 = np.log(Pr_S1)
ln_Pr_S0 = np.log(Pr_S0)

print("Log probability email is spam: " + str(ln_Pr_S1))
print("Log probability email is not spam: " + str(ln_Pr_S0))

ln_Pr_W1_S1 = np.log(np.mean(X_train[y_train == 1], axis = 0))
ln_Pr_W1_S0 = np.log(np.mean(X_train[y_train == 0], axis = 0))

ln_Pr_W1_S1[ln_Pr_W1_S1 == float('-inf')] = -99999999999
ln_Pr_W1_S0[ln_Pr_W1_S0 == float('-inf')] = -99999999999

lnL_S1_W = ln_Pr_S1 + ln_Pr_W1_S1 @ X_test.T
lnL_S0_W = ln_Pr_S0 + ln_Pr_W1_S0 @ X_test.T

y_pred = []
for i in range(len(X_test)):
    if lnL_S1_W[0, i] > lnL_S0_W[0, i]:
        y_pred.append(1)
    elif lnL_S0_W[0, i] > lnL_S1_W[0, i]:
        y_pred.append(0)
    else:
        y_pred.append("tie")
        
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix:")
print(cm)
print("TP: " + str(tp))
print("TN: " + str(tn))
print("FP: " + str(fp))
print("FN: " + str(fn))
print("Accuracy: " + str(100*accuracy_score(y_test, y_pred).round(4)) + " %")
print("Precision: " + str(100*precision_score(y_test, y_pred).round(4)) + " %")
print("Recall: " + str(100*recall_score(y_test, y_pred).round(4)) + " %")

def fitData(alpha):
    X_train, X_test, y_train, y_test = train_test_split(np.matrix(X), y, test_size=0.2, random_state=25)
    
    numSpam = len(X_train[y_train == 1]) 
    numNonSpam = len(X_train[y_train == 0])
    total = len(X_train)

    # Using natural log likelihood
    ln_Pr_S1 = np.log(numSpam / total)
    ln_Pr_S0 = np.log(numNonSpam / total)
    
    ln_Pr_W1_S1 = np.log((X_train[y_train == 1].sum(axis=0) + alpha) / (len(X_train[y_train == 1]) + 2*alpha))
    ln_Pr_W1_S0 = np.log((X_train[y_train == 0].sum(axis=0) + alpha) / (len(X_train[y_train == 0]) + 2*alpha))
    
    lnL_S1_W = ln_Pr_S1 + ln_Pr_W1_S1 @ X_test.T
    lnL_S0_W = ln_Pr_S0 + ln_Pr_W1_S0 @ X_test.T
    
    return (lnL_S0_W, lnL_S1_W, alpha)

    
def predictCategory(logLiks):
    lnL_S0_W, lnL_S1_W, alpha = logLiks
    
    y_pred = []
    for i in range(len(X_test)):
        if lnL_S1_W[0, i] > lnL_S0_W[0, i]:
            y_pred.append(1)
        elif lnL_S0_W[0, i] > lnL_S1_W[0, i]:
            y_pred.append(0)
        else:
            y_pred.append("tie")
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("Alpha value used: " + str(alpha))
    print("Confusion Matrix:")
    print(cm)
    print("TP: " + str(tp))
    print("TN: " + str(tn))
    print("FP: " + str(fp))
    print("FN: " + str(fn))
    print("Accuracy: " + str(100*accuracy_score(y_test, y_pred).round(4)) + " %")
    print("Precision: " + str(100*precision_score(y_test, y_pred).round(4)) + " %")
    print("Recall: " + str(100*recall_score(y_test, y_pred).round(4)) + " %")
    print()
    print("----------------------------------------------------------------------------")
    print()
    
def predictCategoryAlpha(logLiks):
    lnL_S0_W, lnL_S1_W, alpha = logLiks
    
    y_pred = []
    for i in range(len(X_test)):
        if lnL_S1_W[0, i] > lnL_S0_W[0, i]:
            y_pred.append(1)
        elif lnL_S0_W[0, i] > lnL_S1_W[0, i]:
            y_pred.append(0)
        else:
            y_pred.append("tie")
    return (100*accuracy_score(y_test, y_pred).round(4), alpha)
    
    
# Predicting with various alpha values
predictCategory(fitData(0.9))
predictCategory(fitData(0.5))
predictCategory(fitData(0.1))
predictCategory(fitData(0.01))

alphas = [10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1] 
stats = []
for alpha in alphas:
    stats.append(predictCategoryAlpha(fitData(alpha)))
    
print("The best accuracy found was " + str(max(stats)[0]) + "% with an alpha value of " + str(max(stats)[1]))
print()
print()

# Analyzing word predictors for spam & non-spam:

Pr_S1 = len(X[y == 1]) / len(X)
Pr_S0 = len(X[y == 0]) / len(X)

Pr_W1_S1 = (X[y == 1].sum(axis=0)) / len(X[y == 1])
Pr_W1_S0 = (X[y == 0].sum(axis=0)) / len(X[y == 0])

Pr_W1 = X.sum(axis=0) / len(X)

Pr_S1_W1 = (Pr_W1_S1 * Pr_S1) / len(Pr_W1)
Pr_S0_W1 = (Pr_W1_S0 * Pr_S0) / len(Pr_W1)

# Pr_S1_W1[Pr_S1_W1 == 0] = 0.0001
# Pr_S0_W1[Pr_S0_W1 == 0] = 0.0001

predictions_spam = np.log(Pr_S1_W1) - np.log(Pr_S0_W1)
predictions_ns = np.log(Pr_S0_W1) - np.log(Pr_S1_W1)

Pr_S1_W1[Pr_S1_W1 == 0] = 0.0001
Pr_S0_W1[Pr_S0_W1 == 0] = 0.0001

predictions_smoothed_spam = np.log(Pr_S1_W1) - np.log(Pr_S0_W1)
predictions_smoothed_ns = np.log(Pr_S0_W1) - np.log(Pr_S1_W1)

top10SpamIdx = (-predictions_spam).argsort()[:10]
top10NonSpamIdx = (-predictions_ns).argsort()[:10]

top10SmoothedSpamIdx = (-predictions_smoothed_spam).argsort()[:10]
top10SmoothedNonSpamIdx = (-predictions_smoothed_ns).argsort()[:10]

top10Spam = [np.array(vocabulary)[top10SpamIdx.astype(int)]]
top10NonSpam = [np.array(vocabulary)[top10NonSpamIdx.astype(int)]]

top10SmoothedSpam = [np.array(vocabulary)[top10SmoothedSpamIdx.astype(int)]]
top10SmoothedNonSpam = [np.array(vocabulary)[top10SmoothedNonSpamIdx.astype(int)]]

print()
print()
print("Top 10 spam predictors: ")
print()
print('\n'.join(map(str, top10Spam[0])))
print()


# K-fold Cross Validation:
    
def kFoldCV(k, alpha):
    # (a) randomizing
    data = df.sample(frac=1)
    X = vectorizer.fit_transform(df.message).toarray()
    y = data['spam'].to_numpy()*1
    
    # (b) splitting
    Xchunks = np.array_split(X, 4)
    ychunks = np.array_split(y, 4)
    
    # (c) test/train allocation
    Xv = Xchunks[0]
    yv = ychunks[0]
    
    # (d) training
    stats = []
    for Xt, yt in zip(Xchunks[1:], ychunks[1:]):
        
        numSpam = len(Xt[yt == 1]) 
        numNonSpam = len(Xt[yt == 0])
        total = len(Xt)

        ln_Pr_S1 = np.log(numSpam / total)
        ln_Pr_S0 = np.log(numNonSpam / total)

        ln_Pr_W1_S1 = np.log((Xt[yt == 1].sum(axis=0) + alpha) / (len(Xt[yt == 1]) + 2*alpha))
        ln_Pr_W1_S0 = np.log((Xt[yt == 0].sum(axis=0) + alpha) / (len(Xt[yt == 0]) + 2*alpha))

        lnL_S1_W = ln_Pr_S1 + ln_Pr_W1_S1 @ Xv.T
        lnL_S0_W = ln_Pr_S0 + ln_Pr_W1_S0 @ Xv.T
    
        # (e) accuracy & scoring
        y_pred = []
        for i in range(len(Xv)):
            if lnL_S1_W[i] > lnL_S0_W[i]:
                y_pred.append(1)
            elif lnL_S0_W[i] > lnL_S1_W[i]:
                y_pred.append(0)
            else:
                y_pred.append("tie")

        stats.append(100*accuracy_score(yv, y_pred).round(4))
    return np.mean(stats)


alphas = [10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0]
accs = []
for alpha in [10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0]:
    accs.append(kFoldCV(5, alpha))
    
import matplotlib.pyplot as plt
plt.plot(alphas, accs)
plt.xlabel("Alpha values")
plt.xlim(0.0, 0.05)
plt.ylabel("Accuracy (%)")
plt.title("Alpha vs. Accuracy")
print()