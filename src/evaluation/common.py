from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score


def evaluate(actual, predicted, scores) -> dict:
    return{
        'auc': roc_auc_score(actual, scores),
        'f1_score': f1_score(actual, predicted, average='macro'),
        'accuracy': accuracy_score(actual, predicted),
        'precision': precision_score(actual, predicted),
        'recall': recall_score(actual, predicted)
    }
