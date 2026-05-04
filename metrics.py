import numpy as np
np.random.seed(0)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Compute quality metrics for the model
def compute_metrics(labels, preds, current_metrics= [0.0, 0.0, 0.0, 0.0]):
    
    loss, f1, acc, gmean = current_metrics

    if isinstance(labels,np.ndarray):
        running_f1 =  f1_score(labels, preds, zero_division=0)
        running_acc = accuracy_score(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels= [0,1]).ravel()
    else:
        running_f1 =  f1_score(labels.detach().numpy(), preds.detach().numpy(), zero_division=0)
        running_acc = accuracy_score(labels.detach().numpy(), preds.detach().numpy())
        tn, fp, fn, tp = confusion_matrix(labels.detach().numpy(), preds.detach().numpy(), labels= [0,1]).ravel()
    
    sensitivity = tp/(tp+fn) if (tp+fn != 0) else 0
    specificity = tn/(tn+fp) if(tn+fp != 0) else 0 
    running_gmean = np.sqrt(sensitivity*specificity) 

    f1 = f1 + running_f1
    acc = acc + running_acc
    gmean = gmean + running_gmean

    return [loss, f1, acc, gmean]
