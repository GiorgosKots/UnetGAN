# import matplotlib.pyplot as plt
# import numpy as np

# def plot_jsd(iteration_losses, jsd_history, total_iterations, num_epochs, dataloader_size):
#     plt.figure(figsize=(12, 16))

#     # First subplot for iteration losses
#     plt.subplot(4, 1, 1)
#     iterations = range(total_iterations)
#     plt.plot(iterations, iteration_losses['total_d_loss'], label='D Loss', color='blue')
#     plt.plot(iterations, iteration_losses['total_g_loss'], label='G Loss', color='red')
#     plt.title('Generator and Discriminator Losses per Iteration')
#     plt.xlabel('Iteration')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     # Second subplot for epoch losses
#     plt.subplot(4, 1, 2)
#     # Calculate average loss per epoch
#     epochs = range(num_epochs)
#     d_losses_per_epoch = []
#     g_losses_per_epoch = []

#     for epoch in epochs:
#         start_idx = epoch * dataloader_size
#         end_idx = (epoch + 1) * dataloader_size

#         d_epoch_loss = np.mean(iteration_losses['total_d_loss'][start_idx:end_idx])
#         g_epoch_loss = np.mean(iteration_losses['total_g_loss'][start_idx:end_idx])

#         d_losses_per_epoch.append(d_epoch_loss)
#         g_losses_per_epoch.append(g_epoch_loss)

#     plt.plot(epochs, d_losses_per_epoch, label='D Loss', color='blue')
#     plt.plot(epochs, g_losses_per_epoch, label='G Loss', color='red')
#     plt.title('Generator and Discriminator Losses per Epoch')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     # Third subplot for WGAN-specific losses
#     plt.subplot(4, 1, 3)
#     gradient_penalty_per_epoch = []
#     dec_losses_per_epoch = []

#     for epoch in epochs:
#         start_idx = epoch * dataloader_size
#         end_idx = (epoch + 1) * dataloader_size

#         gp_epoch_loss = np.mean(iteration_losses['gradient_penalty'][start_idx:end_idx])
#         dec_epoch_loss = np.mean(iteration_losses['dec_loss'][start_idx:end_idx])

#         gradient_penalty_per_epoch.append(gp_epoch_loss)
#         dec_losses_per_epoch.append(dec_epoch_loss)

#     plt.plot(epochs, gradient_penalty_per_epoch, label='Gradient Penalty', color='purple')
#     plt.plot(epochs, dec_losses_per_epoch, label='Decoder Loss', color='orange')
#     plt.title('WGAN-GP Specific Losses per Epoch')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     # Fourth subplot for JSD with fixed y-axis
#     plt.subplot(4, 1, 4)
#     jsd_epochs = range(len(jsd_history))
#     plt.plot(jsd_epochs, jsd_history, 'g-', label='JSD Score')
#     plt.title('JSD Score Progress')
#     plt.xlabel('Epoch')
#     plt.ylabel('JSD Score')
#     plt.ylim(0, 1)  # Set fixed y-axis range
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()

# def plot_jsd_fred(iteration_losses, jsd_history, fred_history, total_iterations, num_epochs, dataloader_size):
#     plt.figure(figsize=(12, 20))  # Increased height to accommodate 5 subplots

#     # First subplot for iteration losses
#     plt.subplot(5, 1, 1)
#     iterations = range(total_iterations)
#     plt.plot(iterations, iteration_losses['total_d_loss'], label='D Loss', color='blue')
#     plt.plot(iterations, iteration_losses['total_g_loss'], label='G Loss', color='red')
#     plt.title('Generator and Discriminator Losses per Iteration')
#     plt.xlabel('Iteration')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     # Second subplot for epoch losses
#     plt.subplot(5, 1, 2)
#     # Calculate average loss per epoch
#     epochs = range(num_epochs)
#     d_losses_per_epoch = []
#     g_losses_per_epoch = []

#     for epoch in epochs:
#         start_idx = epoch * dataloader_size
#         end_idx = (epoch + 1) * dataloader_size

#         d_epoch_loss = np.mean(iteration_losses['total_d_loss'][start_idx:end_idx])
#         g_epoch_loss = np.mean(iteration_losses['total_g_loss'][start_idx:end_idx])

#         d_losses_per_epoch.append(d_epoch_loss)
#         g_losses_per_epoch.append(g_epoch_loss)

#     plt.plot(epochs, d_losses_per_epoch, label='D Loss', color='blue')
#     plt.plot(epochs, g_losses_per_epoch, label='G Loss', color='red')
#     plt.title('Generator and Discriminator Losses per Epoch')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     # Third subplot for WGAN-specific losses
#     plt.subplot(5, 1, 3)
#     gradient_penalty_per_epoch = []
#     dec_losses_per_epoch = []

#     for epoch in epochs:
#         start_idx = epoch * dataloader_size
#         end_idx = (epoch + 1) * dataloader_size

#         gp_epoch_loss = np.mean(iteration_losses['gradient_penalty'][start_idx:end_idx])
#         dec_epoch_loss = np.mean(iteration_losses['dec_loss'][start_idx:end_idx])

#         gradient_penalty_per_epoch.append(gp_epoch_loss)
#         dec_losses_per_epoch.append(dec_epoch_loss)

#     plt.plot(epochs, gradient_penalty_per_epoch, label='Gradient Penalty', color='purple')
#     plt.plot(epochs, dec_losses_per_epoch, label='Decoder Loss', color='orange')
#     plt.title('WGAN-GP Specific Losses per Epoch')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     # Fourth subplot for JSD with fixed y-axis
#     plt.subplot(5, 1, 4)
#     jsd_epochs = range(len(jsd_history))
#     plt.plot(jsd_epochs, jsd_history, 'g-', label='JSD Score')
#     plt.title('JSD Score Progress')
#     plt.xlabel('Epoch')
#     plt.ylabel('JSD Score')
#     plt.ylim(0, 1)  # Set fixed y-axis range
#     plt.legend()
#     plt.grid(True)

#     # Fifth subplot for FReD
#     plt.subplot(5, 1, 5)
#     fred_epochs = range(len(fred_history))
#     plt.plot(fred_epochs, fred_history, 'm-', label='FReD Score')
#     plt.title('FReD Score Progress')
#     plt.xlabel('Epoch')
#     plt.ylabel('FReD Score')
#     plt.ylim(0, 5)  # Adjust this based on your actual FReD values
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_jsd(iteration_losses, jsd_history, total_iterations, num_epochs, dataloader_size):
    plt.figure(figsize=(12, 20))  # Increased height for 5 subplots

    # First subplot for iteration losses with gradient penalty
    plt.subplot(5, 1, 1)
    iterations = range(total_iterations)
    plt.plot(iterations, iteration_losses['total_d_loss'], label='D Loss', color='blue')
    plt.plot(iterations, iteration_losses['total_g_loss'], label='G Loss', color='red')
    plt.plot(iterations, iteration_losses['gradient_penalty'], ':', label='Gradient Penalty', color='purple')
    plt.title('Generator, Discriminator, and Gradient Penalty Losses per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Second subplot for epoch losses with gradient penalty
    plt.subplot(5, 1, 2)
    epochs = range(num_epochs)
    d_losses_per_epoch = []
    g_losses_per_epoch = []
    gp_losses_per_epoch = []

    for epoch in epochs:
        start_idx = epoch * dataloader_size
        end_idx = (epoch + 1) * dataloader_size

        d_epoch_loss = np.mean(iteration_losses['total_d_loss'][start_idx:end_idx])
        g_epoch_loss = np.mean(iteration_losses['total_g_loss'][start_idx:end_idx])
        gp_epoch_loss = np.mean(iteration_losses['gradient_penalty'][start_idx:end_idx])

        d_losses_per_epoch.append(d_epoch_loss)
        g_losses_per_epoch.append(g_epoch_loss)
        gp_losses_per_epoch.append(gp_epoch_loss)

    plt.plot(epochs, d_losses_per_epoch, label='D Loss', color='blue')
    plt.plot(epochs, g_losses_per_epoch, label='G Loss', color='red')
    plt.plot(epochs, gp_losses_per_epoch, ':', label='Gradient Penalty', color='purple')
    plt.title('Generator, Discriminator, and Gradient Penalty Losses per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Third subplot for decoder loss
    plt.subplot(5, 1, 3)
    dec_losses_per_epoch = []

    for epoch in epochs:
        start_idx = epoch * dataloader_size
        end_idx = (epoch + 1) * dataloader_size
        dec_epoch_loss = np.mean(iteration_losses['dec_loss'][start_idx:end_idx])
        dec_losses_per_epoch.append(dec_epoch_loss)

    plt.plot(epochs, dec_losses_per_epoch, label='Decoder Loss', color='orange')
    plt.title('Decoder Loss per Epoch')
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
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_jsd_fred(iteration_losses, jsd_history, fred_history, total_iterations, num_epochs, dataloader_size):
    plt.figure(figsize=(12, 24))  # Increased height for 6 subplots

    # First subplot for iteration losses with gradient penalty
    plt.subplot(6, 1, 1)
    iterations = range(total_iterations)
    plt.plot(iterations, iteration_losses['total_d_loss'], label='D Loss', color='blue')
    plt.plot(iterations, iteration_losses['total_g_loss'], label='G Loss', color='red')
    plt.plot(iterations, iteration_losses['gradient_penalty'], ':', label='Gradient Penalty', color='purple')
    plt.title('Generator, Discriminator, and Gradient Penalty Losses per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Second subplot for epoch losses with gradient penalty
    plt.subplot(6, 1, 2)
    epochs = range(num_epochs)
    d_losses_per_epoch = []
    g_losses_per_epoch = []
    gp_losses_per_epoch = []

    for epoch in epochs:
        start_idx = epoch * dataloader_size
        end_idx = (epoch + 1) * dataloader_size

        d_epoch_loss = np.mean(iteration_losses['total_d_loss'][start_idx:end_idx])
        g_epoch_loss = np.mean(iteration_losses['total_g_loss'][start_idx:end_idx])
        gp_epoch_loss = np.mean(iteration_losses['gradient_penalty'][start_idx:end_idx])

        d_losses_per_epoch.append(d_epoch_loss)
        g_losses_per_epoch.append(g_epoch_loss)
        gp_losses_per_epoch.append(gp_epoch_loss)

    plt.plot(epochs, d_losses_per_epoch, label='D Loss', color='blue')
    plt.plot(epochs, g_losses_per_epoch, label='G Loss', color='red')
    plt.plot(epochs, gp_losses_per_epoch, ':', label='Gradient Penalty', color='purple')
    plt.title('Generator, Discriminator, and Gradient Penalty Losses per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Third subplot for decoder loss
    plt.subplot(6, 1, 3)
    dec_losses_per_epoch = []

    for epoch in epochs:
        start_idx = epoch * dataloader_size
        end_idx = (epoch + 1) * dataloader_size
        dec_epoch_loss = np.mean(iteration_losses['dec_loss'][start_idx:end_idx])
        dec_losses_per_epoch.append(dec_epoch_loss)

    plt.plot(epochs, dec_losses_per_epoch, label='Decoder Loss', color='orange')
    plt.title('Decoder Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Fourth subplot for JSD with fixed y-axis
    plt.subplot(6, 1, 4)
    jsd_epochs = range(len(jsd_history))
    plt.plot(jsd_epochs, jsd_history, 'g-', label='JSD Score')
    plt.title('JSD Score Progress')
    plt.xlabel('Epoch')
    plt.ylabel('JSD Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    # Fifth subplot for FReD
    plt.subplot(6, 1, 5)
    fred_epochs = range(len(fred_history))
    plt.plot(fred_epochs, fred_history, 'm-', label='FReD Score')
    plt.title('FReD Score Progress')
    plt.xlabel('Epoch')
    plt.ylabel('FReD Score')
    plt.ylim(0, 5)  # Adjust based on your actual FReD values
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()