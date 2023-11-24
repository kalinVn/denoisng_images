# Build denoise auto encoder that reconstruct noisy image documents
# using keras sequential model and conv2d layers

from service import NN_Conv_Denois_Encoder
from visualizator import plot_random_digits_after_fit


params = {
    'saved_model_name': 'conv_saved_train_model_noise_documents',
    'preprocess_name': 'noisy_documents',
    'model_name': 'conv_denoise_autoencoder',
    'optimizer': 'Adam',
    'loss': 'binary_crossentropy',
    'features_dimension': (420, 540, 1),
    'layers': [
        {
            'args': {
                'filters': 8,
                'kernel_size': (3, 3),
                'activation': 'relu',
                'padding': 'same',
                'input_shape': (420, 540, 1)
            },
            'type': 'conv2D'
        },
        {
            'args': {
                'filters': 16,
                'kernel_size': (3, 3),
                'activation': 'relu',
                'padding': 'same'
            },
            'type': 'conv2D'
        },
        {
            'args': {
                'filters': 1,
                'kernel_size': (3, 3),
                'activation': 'sigmoid',
                'padding': 'same',
            },

            'type': 'conv2D'
        }
    ]
}
nn_denoise_autoencoder = NN_Conv_Denois_Encoder(**params)
nn_denoise_autoencoder.set_train_test_data()
nn_denoise_autoencoder.fit()
x_train_noisy = nn_denoise_autoencoder.get_preprocess().get_x_train_noisy()
x_train_clean = nn_denoise_autoencoder.get_preprocess().get_x_train_clean()

params = {

    'cols': 3,
    'rows': 2,
    'reshape': [420, 540, -1],
    'reshape_size': {
        'width': 420,
        'height': 540
    },
    'figsize': {
        'width': 20,
        'height': 7
    },
    'plots_data': [x_train_noisy, x_train_clean],
    'random_img_range': 50
}

plot_random_digits_after_fit(params)
