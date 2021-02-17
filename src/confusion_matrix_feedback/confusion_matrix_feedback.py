import numpy as np
from pymining import itemmining

from src.predictive_model.predictive_model import drop_columns


def compute_feedback(explanations, predictive_model, feedback_df, encoder, threshold=None, top_k=None):
    predicted = predictive_model.model.predict(drop_columns(feedback_df))
    actual = feedback_df['label']

    trace_ids = feedback_df['trace_id']

    confusion_matrix = _retrieve_confusion_matrix_ids(trace_ids, predicted, actual, encoder)

    filtered_explanations = _filter_explanations(explanations, confusion_matrix, threshold=threshold)

    frequent_patterns = _mine_frequent_patterns(confusion_matrix, filtered_explanations)

    feedback = {
        'true': _subtract_patterns(frequent_patterns['fp'], frequent_patterns['tp']),
        'false': _subtract_patterns(frequent_patterns['fn'], frequent_patterns['tn'])
    }

    feedback = {
        classes: sorted(feedback[classes], key=lambda x: x[1] * len(x[0]) / len(feedback[classes]), reverse=True)
        for classes in feedback
    }

    if top_k is not None:
        for classes in feedback:
            feedback[classes] = feedback[classes][:top_k]

    return feedback


def _retrieve_confusion_matrix_ids(trace_ids, predicted, actual, encoder) -> dict:
    decoded_predicted = encoder.decode_column(predicted, 'label')
    decoded_actual = encoder.decode_column(actual, 'label')
    elements = np.column_stack((
        trace_ids,
        decoded_predicted,
        decoded_actual
    )).tolist()

    tp = {
        trace_id
        for trace_id, predicted, actual in elements
        if actual == 'true' and predicted == 'true'
    }
    tn = {
        trace_id
        for trace_id, predicted, actual in elements
        if actual == 'false' and predicted == 'false'
    }
    fp = {
        trace_id
        for trace_id, predicted, actual in elements
        if actual == 'false' and predicted == 'true'
    }
    fn = {
        trace_id
        for trace_id, predicted, actual in elements
        if actual == 'true' and predicted == 'false'
    }
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def _set_threshold(explanations, confusion_matrix, threshold=None):
    trace_ids = {
        'true': list(confusion_matrix['tp']) + list(confusion_matrix['fp']),
        'false': list(confusion_matrix['tn']) + list(confusion_matrix['fn'])
    }

    values = {
        'true': [feature[2] for tid in trace_ids['true'] for feature in explanations[tid] if feature[2] > 0],
        'false': [-feature[2] for tid in trace_ids['false'] for feature in explanations[tid] if feature[2] < 0]
    }

    if threshold is None:
        threshold = {classes: 0 for classes in trace_ids}

    while(True):
        filtered_explanations = _filter_explanations(explanations, confusion_matrix, threshold=threshold)
        if any(len(filtered_explanations[trace]) > 20 for trace in filtered_explanations) :
            for classes in trace_ids:
                if any(len(filtered_explanations[trace]) > 20 for trace in trace_ids[classes]):
                    threshold[classes] = np.mean(values[classes])
                    values[classes] = list(filter(lambda x: x > threshold[classes], values[classes]))
        else:
            return threshold


def _filter_explanations(explanations, confusion_matrix, threshold=None):
    true = list(confusion_matrix['tp']) + list(confusion_matrix['fp'])
    false = list(confusion_matrix['tn']) + list(confusion_matrix['fn'])

    filtered_explanations = {}
    for trace_id in explanations:
        filtered_explanations[trace_id] = []

        if threshold is None:
            threshold = 12
        else:
            threshold = min(12, threshold)

        thresholds = {}
        if trace_id in true:
            thresholds['true'] = sorted(explanations[trace_id], key=lambda x: x[2], reverse=True)[min(len(explanations[trace_id]), threshold)][2]
        elif trace_id in false:
            thresholds['false'] = - sorted(explanations[trace_id], key=lambda x: -x[2], reverse=True)[min(len(explanations[trace_id]), threshold)][2]

        for feature_name, value, importance in explanations[trace_id]:
            if trace_id in true and importance > thresholds['true']:  # this becomes a for each in multiclass
                filtered_explanations[trace_id] += [ (feature_name, value, importance) ]
            elif trace_id in false and -importance > thresholds['false']:
                filtered_explanations[trace_id] += [ (feature_name, value, importance) ]

    return filtered_explanations


def _mine_frequent_patterns(confusion_matrix, filtered_explanations):
    mined_patterns = {}
    for element in confusion_matrix:
        mined_patterns[element] = itemmining.relim(itemmining.get_relim_input([
            [
                str(feature_name) + '//' + str(value) # + '_' + str(_tassellate_number(importance))
                for feature_name, value, importance in filtered_explanations[tid]
            ]
            for tid in confusion_matrix[element]
        ]), min_support=2)
        mined_patterns[element] = sorted(
            [
                ([ el.split('//') for el in list(key) ], mined_patterns[element][key])
                for key in mined_patterns[element]
            ],
            key=lambda x: x[1],
            reverse=True
        )

    return mined_patterns


def _tassellate_number(element):
    element = str(element).split('.')
    return element[0] + '.' + element[1][:3]


def _subtract_patterns(list1, list2):

    difference = list1
    for el, _ in list2:
        if el in [e[0] for e in difference]:
            index = [e[0] for e in difference].index(el)
            difference.remove(difference[index])

    return difference

