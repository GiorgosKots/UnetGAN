import matplotlib.pyplot as plt
import numpy as np

def plot_jsd(iteration_losses, jsd_history, total_iterations, num_epochs, dataloader_size):
    plt.figure(figsize=(12, 16))  # Adjusted figure size to accommodate the additional subplot

    # First subplot for iteration losses
    plt.subplot(4, 1, 1)
    iterations = range(total_iterations)
    plt.plot(iterations, iteration_losses['total_d_loss'], label='D Loss', color='blue')
    plt.plot(iterations, iteration_losses['total_g_loss'], label='G Loss', color='red')
    plt.title('Generator and Discriminator Losses per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Second subplot for epoch losses
    plt.subplot(4, 1, 2)
    # Calculate average loss per epoch
    epochs = range(num_epochs)
    d_losses_per_epoch = []
    g_losses_per_epoch = []

    for epoch in epochs:
        start_idx = epoch * dataloader_size
        end_idx = (epoch + 1) * dataloader_size

        d_epoch_loss = np.mean(iteration_losses['total_d_loss'][start_idx:end_idx])
        g_epoch_loss = np.mean(iteration_losses['total_g_loss'][start_idx:end_idx])

        d_losses_per_epoch.append(d_epoch_loss)
        g_losses_per_epoch.append(g_epoch_loss)

    plt.plot(epochs, d_losses_per_epoch, label='D Loss', color='blue')
    plt.plot(epochs, g_losses_per_epoch, label='G Loss', color='red')
    plt.title('Generator and Discriminator Losses per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Third subplot for detailed losses
    plt.subplot(4, 1, 3)
    enc_losses_per_epoch = []
    dec_losses_per_epoch = []
    g_global_losses_per_epoch = []
    g_pixel_losses_per_epoch = []
    d_total_losses_per_epoch = []
    g_total_losses_per_epoch = []

    for epoch in epochs:
        start_idx = epoch * dataloader_size
        end_idx = (epoch + 1) * dataloader_size

        enc_epoch_loss = np.mean(iteration_losses['enc_loss'][start_idx:end_idx])
        dec_epoch_loss = np.mean(iteration_losses['dec_loss'][start_idx:end_idx])
        g_global_epoch_loss = np.mean(iteration_losses['g_global_loss'][start_idx:end_idx])
        g_pixel_epoch_loss = np.mean(iteration_losses['g_pixel_loss'][start_idx:end_idx])
        d_total_epoch_loss = np.mean(iteration_losses['total_d_loss'][start_idx:end_idx])
        g_total_epoch_loss = np.mean(iteration_losses['total_g_loss'][start_idx:end_idx])

        enc_losses_per_epoch.append(enc_epoch_loss)
        dec_losses_per_epoch.append(dec_epoch_loss)
        g_global_losses_per_epoch.append(g_global_epoch_loss)
        g_pixel_losses_per_epoch.append(g_pixel_epoch_loss)
        d_total_losses_per_epoch.append(d_total_epoch_loss)
        g_total_losses_per_epoch.append(g_total_epoch_loss)

    plt.plot(epochs, enc_losses_per_epoch, label='Enc Loss', color='purple')
    plt.plot(epochs, dec_losses_per_epoch, label='Dec Loss', color='orange')
    plt.plot(epochs, g_global_losses_per_epoch, label='G Global Loss', color='green')
    plt.plot(epochs, g_pixel_losses_per_epoch, label='G Pixel Loss', color='brown')
    plt.plot(epochs, d_total_losses_per_epoch, label='D Total Loss', color='blue', linestyle='--')
    plt.plot(epochs, g_total_losses_per_epoch, label='G Total Loss', color='red', linestyle='--')
    plt.title('Detailed Losses per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Fourth subplot for JSD with fixed y-axis
    plt.subplot(4, 1, 4)
    jsd_epochs = range(len(jsd_history))
    plt.plot(jsd_epochs, jsd_history, 'g-', label='JSD Score')
    plt.title('JSD Score Progress')
    plt.xlabel('Epoch')
    plt.ylabel('JSD Score')
    plt.ylim(0, 1)  # Set fixed y-axis range
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_jsd_fred(iteration_losses, jsd_history, fred_history, total_iterations, num_epochs, dataloader_size):
    plt.figure(figsize=(12, 20))  # Increased height to accommodate 5 subplots

    # First subplot for iteration losses
    plt.subplot(5, 1, 1)
    iterations = range(total_iterations)
    plt.plot(iterations, iteration_losses['total_d_loss'], label='D Loss', color='blue')
    plt.plot(iterations, iteration_losses['total_g_loss'], label='G Loss', color='red')
    plt.title('Generator and Discriminator Losses per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Second subplot for epoch losses
    plt.subplot(5, 1, 2)
    # Calculate average loss per epoch
    epochs = range(num_epochs)
    d_losses_per_epoch = []
    g_losses_per_epoch = []

    for epoch in epochs:
        start_idx = epoch * dataloader_size
        end_idx = (epoch + 1) * dataloader_size

        d_epoch_loss = np.mean(iteration_losses['total_d_loss'][start_idx:end_idx])
        g_epoch_loss = np.mean(iteration_losses['total_g_loss'][start_idx:end_idx])

        d_losses_per_epoch.append(d_epoch_loss)
        g_losses_per_epoch.append(g_epoch_loss)

    plt.plot(epochs, d_losses_per_epoch, label='D Loss', color='blue')
    plt.plot(epochs, g_losses_per_epoch, label='G Loss', color='red')
    plt.title('Generator and Discriminator Losses per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Third subplot for detailed losses
    plt.subplot(5, 1, 3)
    enc_losses_per_epoch = []
    dec_losses_per_epoch = []
    g_global_losses_per_epoch = []
    g_pixel_losses_per_epoch = []
    d_total_losses_per_epoch = []
    g_total_losses_per_epoch = []

    for epoch in epochs:
        start_idx = epoch * dataloader_size
        end_idx = (epoch + 1) * dataloader_size

        enc_epoch_loss = np.mean(iteration_losses['enc_loss'][start_idx:end_idx])
        dec_epoch_loss = np.mean(iteration_losses['dec_loss'][start_idx:end_idx])
        g_global_epoch_loss = np.mean(iteration_losses['g_global_loss'][start_idx:end_idx])
        g_pixel_epoch_loss = np.mean(iteration_losses['g_pixel_loss'][start_idx:end_idx])
        d_total_epoch_loss = np.mean(iteration_losses['total_d_loss'][start_idx:end_idx])
        g_total_epoch_loss = np.mean(iteration_losses['total_g_loss'][start_idx:end_idx])

        enc_losses_per_epoch.append(enc_epoch_loss)
        dec_losses_per_epoch.append(dec_epoch_loss)
        g_global_losses_per_epoch.append(g_global_epoch_loss)
        g_pixel_losses_per_epoch.append(g_pixel_epoch_loss)
        d_total_losses_per_epoch.append(d_total_epoch_loss)
        g_total_losses_per_epoch.append(g_total_epoch_loss)

    plt.plot(epochs, enc_losses_per_epoch, label='Enc Loss', color='purple')
    plt.plot(epochs, dec_losses_per_epoch, label='Dec Loss', color='orange')
    plt.plot(epochs, g_global_losses_per_epoch, label='G Global Loss', color='green')
    plt.plot(epochs, g_pixel_losses_per_epoch, label='G Pixel Loss', color='brown')
    plt.plot(epochs, d_total_losses_per_epoch, label='D Total Loss', color='blue', linestyle='--')
    plt.plot(epochs, g_total_losses_per_epoch, label='G Total Loss', color='red', linestyle='--')
    plt.title('Detailed Losses per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Fourth subplot for JSD with fixed y-axis
    plt.subplot(5, 1, 4)
    jsd_epochs = range(len(jsd_history))
    plt.plot(jsd_epochs, jsd_history, 'g-', label='JSD Score')
    plt.title('JSD Score Progress')
    plt.xlabel('Epoch')
    plt.ylabel('JSD Score')
    plt.ylim(0, 1)  # Set fixed y-axis range
    plt.legend()
    plt.grid(True)

    # Fifth subplot for FReD
    plt.subplot(5, 1, 5)
    fred_epochs = range(len(fred_history))
    plt.plot(fred_epochs, fred_history, 'm-', label='FReD Score')
    plt.title('FReD Score Progress')
    plt.xlabel('Epoch')
    plt.ylabel('FReD Score')
    # Set y-axis range based on your observed FReD values
    # For example, if your FReD scores range between 0-100:
    plt.ylim(0, 100)  # Adjust this based on your actual FReD values
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage:
# from plot_metrics import plot_jsd, plot_jsd_fred
# plot_jsd(iteration_losses, jsd_history, total_iterations, num_epochs, dataloader_size)
# or
#plot_jsd_fred(iteration_losses, jsd_history, fred_history, total_iterations, num_epochs, dataloader_size)