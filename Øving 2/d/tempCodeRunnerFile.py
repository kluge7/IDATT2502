for i in range(10):
    weight_image = weights[i, :].reshape(28, 28)
    plt.imshow(weight_image, cmap='viridis')
    plt.title(f'Weights for digit {i}')
    plt.colorbar()
    plt.savefig(f'weight_digit_{i}.png')
    plt.close()