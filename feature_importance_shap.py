import pandas
import numpy
import pickle
import shap
import modelop.utils as utils

logger = utils.configure_logger()

# modelop.init
def begin():

    global predictive_features
    global shap_explainer

    # Load pickled predictive feature list
    predictive_features = pickle.load(open("predictive_features.pickle", "rb"))
    logger.info("predictive_features: %s", predictive_features)

    # Load pre-trained SHAP explainer
    shap_explainer = pickle.load(open("shap_explainer.pickle", "rb"))
    logger.info("shap_explainer loaded!")


# modelop.metrics
def metrics(dataframe):
    
    # Dictionary to hold final metrics
    metrics = {}

    # Getting dummies for shap values
    data_processed = preprocess(dataframe)[predictive_features]

    # Assigning metrics to output dictionary
    metrics["interpretability"] = [compute_feature_importance(data_processed)]

    yield metrics


def preprocess(data: pandas.DataFrame) -> pandas.DataFrame:
    """[summary]
    TODO
    Args:
        data (pandas.DataFrame): [description]

    Returns:
        pandas.DataFrame: [description]
    """

    data_processed = data.copy(deep=True)
    
    # One-Hot encode data with pd.get_dummies()
    data_processed = pandas.get_dummies(data_processed)

    # In case features don't exist that are needed for the model (possible when dummying)
    # will create columns of zeros for those features
    for col in predictive_features:
        if col not in data_processed.columns:
            data_processed[col] = numpy.zeros(data_processed.shape[0])

    return data_processed


def compute_feature_importance(data:pandas.DataFrame) -> dict:
    """[summary]
    TODO
    Args:
        data (pandas.DataFrame): [description]

    Returns:
        dict: [description]
    """

    # Getting SHAP values
    shap_values = shap_explainer.shap_values(data)

    # Re-organizing and sorting SHAP values
    shap_values = numpy.mean(abs(shap_values), axis=0).tolist()
    shap_values = dict(zip(data.columns, shap_values))
    sorted_shap_values = {
        k: v for k, v in sorted(shap_values.items(), key=lambda x: x[1])
    }

    return {
        "test_name": "SHAP",
        "test_category": "interpretability",
        "test_type": "shap",
        "metric": "feature_importance",
        "test_id": "interpretability_shap_feature_importance",
        "values": sorted_shap_values,
    }
