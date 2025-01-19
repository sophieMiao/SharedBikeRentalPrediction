from lstm.model import LSTMModel
from trans.model import TransformerModel
from mixed.model import MixedModel


def get_model(name, input_dim, prediction_horizon):
    if name == "lstm":
        return LSTMModel(input_size=input_dim, output_size=prediction_horizon)
    elif name == "mix":
        return MixedModel(features_dim=input_dim, output_size=prediction_horizon)
    else:
        return TransformerModel(features_dim=input_dim, output_size=prediction_horizon)
