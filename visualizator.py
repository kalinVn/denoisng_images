from matplotlib import pyplot as plt
import random


def plot_digits(x_test, y_test):
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(10, 5))

    for idx, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]):

        for i in range(2000):
            if y_test[i] == idx:
                ax.imshow(x_test[i], cmap='gray')
                break

    plt.tight_layout()
    plt.show()


def plot_random_digits_after_fit(params):

    fig, axes = plt.subplots(params['rows'], params['cols'], figsize=(params['figsize']['width'], params['figsize']['height']))

    random_images_indexes = random.sample(range(params['random_img_range']), 5)

    for i in range(0, len(params['plots_data'])):
        data = params['plots_data'][i]

        current_axes = axes[i]
        current_index = 0

        for ax in current_axes:
            ax.imshow(data[random_images_indexes[current_index]].reshape(params['reshape']), cmap='gray')
            current_index += 1

    plt.tight_layout()
    plt.show()


    # for data in params.items:
    #     randomly_selected_images = random.sample(range(data.shape[0]), 5)
    #
    #     for index, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    #         ax.imshow(x_test[randomly_selected_images[index]], cmap='gray')
    #
    #         if index == 0:
    #             ax.set_ylabel('INPUT', size=40)
    #
    # # randomly select 5 images
    # randomly_selected_imgs = random.sample(range(x_test.shape[0]), 5)
    #
    # for index, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    #     ax.imshow(x_test[randomly_selected_imgs[index]], cmap='gray')
    #
    #     if index == 0:
    #         ax.set_ylabel('INPUT', size=40)
    #
    # for index, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    #     ax.imshow(output[randomly_selected_imgs[index]].reshape(28, 28), cmap='gray')
    #
    #     if index == 0:
    #         ax.set_ylabel('OUTPUT', size=40)
    #
    # plt.tight_layout()
    # plt.show()
    #




