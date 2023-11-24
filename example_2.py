# Build simple auto encoder that reconstruct mnist digits
# using keras sequential model and Dense layers

from service import NN_Denoise_Autoencoder, NN_Conv_Denois_Encoder, Drive
from visualizator import plot_random_digits_after_fit

params = {
    'saved_model_name': 'base_nn_saved_train_model_digits',
    'preprocess_name': 'mnist_digits',
    'model_name': 'base_denoise_autoencoder',
    'optimizer': 'Adam',
    'loss': 'mean_squared_error',
    'features_dimension': (60000, 28, 28, 1),
    'layers': [
        {
            'args': {
                'units': 32,
                'activation': 'relu',
                'input_shape': (784, ),
            },
            'type': 'dense'
        },
        {
            'args': {
                'units': 32,
                'activation': 'relu',
                'input_shape': (784, ),
            },
            'type': 'dense'
        },

        {
            'args': {
                'units': 32,
                'activation': 'relu',
                'input_shape': (784,),
            },
            'type': 'dense'
        },
{
            'args': {
                'units': 32,
                'activation': 'relu',
                'input_shape': (784, ),
            },
            'type': 'dense'
        },

        {
            'args': {
                'units': 32,
                'activation': 'relu',
                'input_shape': (784,),
            },
            'type': 'dense'
        },
        {
            'args': {
                'units': 784,
                'activation': 'sigmoid',
            },
            'type': 'dense'
        }
    ]
}

nn = NN_Denoise_Autoencoder(**params)

nn.set_train_test_data()
nn.fit()
nn.fit()
nn.predict()
output = nn.get_output()

params = {
    'cols': 5,
    'rows': 2,
    'reshape': (28, 28, 1),
    'random_img_range': 100,
    'figsize': {
        'width': 20,
        'height': 7
    },
    'plots_data': [output,  nn.get_preprocess().get_x_train()]
}


plot_random_digits_after_fit(params)

