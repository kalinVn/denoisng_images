from builder import SequentialModel

class Model:

    @staticmethod
    def get_keras_model(params):
        model_name = params['model_name']

        model = None

        if model_name == 'base_denoise_autoencoder':
            model = SequentialModel.build(params)

        if model_name == 'conv_denoise_autoencoder':
            model = SequentialModel.build(params)

        if model is not None:
            return model

        return None

