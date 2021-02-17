from enum import Enum

from src.explanation.wrappers.lime_wrapper import lime_explain
from src.explanation.wrappers.shap_wrapper import shap_explain


class ExplainerType(Enum):
    SHAP = 'shap'
    LIME = 'lime'


def explain(explainer, predictive_model, test_df, encoder):
    if explainer is ExplainerType.SHAP.value:
        return shap_explain(predictive_model, test_df, encoder)
    elif explainer is ExplainerType.LIME.value:
        return lime_explain(predictive_model, test_df, encoder)
    else:
        raise Exception('selected explainer not yet supported')
