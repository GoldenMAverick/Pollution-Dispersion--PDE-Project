import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm

# Check for GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Simulation parameters
D = 0.1  # Diffusion coefficient (m^2/s)
u = 1.0  # Flow velocity (m/s)
x_min, x_max = -10.0, 10.0  # Spatial domain (m)
t_min, t_max = 0.0, 5.0     # Time domain (s)
N_train = 10000             # Number of training points
N_boundary = 1000           # Number of boundary points
N_initial = 1000            # Number of initial points
epochs = 5000               # Training epochs

# Define the neural network
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh', input_shape=(2,)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        return self.dense_layers(inputs)

# Compute derivatives using automatic differentiation
def compute_derivatives(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        xt = tf.stack([x, t], axis=1)
        C = model(xt)
        C_x = tape.gradient(C, x)
        C_xx = tape.gradient(C_x, x)
        C_t = tape.gradient(C, t)
    del tape
    return C, C_t, C_x, C_xx

# Loss function
def loss_function(model, x_pde, t_pde, x_ic, t_ic, C_ic, x_bc, t_bc, C_bc):
    C, C_t, C_x, C_xx = compute_derivatives(model, x_pde, t_pde)
    pde_residual = C_t - D * C_xx + u * C_x
    pde_loss = tf.reduce_mean(tf.square(pde_residual))

    C_ic_pred = model(tf.stack([x_ic, t_ic], axis=1))
    ic_loss = tf.reduce_mean(tf.square(C_ic_pred - C_ic))

    C_bc_pred = model(tf.stack([x_bc, t_bc], axis=1))
    bc_loss = tf.reduce_mean(tf.square(C_bc_pred - C_bc))

    return pde_loss + ic_loss + bc_loss

# Generate training data
def generate_training_data():
    x_pde = np.random.uniform(x_min, x_max, N_train)
    t_pde = np.random.uniform(t_min, t_max, N_train)

    x_ic = np.random.uniform(x_min, x_max, N_initial)
    t_ic = np.zeros(N_initial)
    C_ic = np.exp(-x_ic**2 / (2 * 0.5**2))

    x_bc = np.concatenate([
        np.full(N_boundary // 2, x_min),
        np.full(N_boundary // 2, x_max)
    ])
    t_bc = np.random.uniform(t_min, t_max, N_boundary)
    C_bc = np.zeros(N_boundary)

    return (
        tf.convert_to_tensor(x_pde, dtype=tf.float32),
        tf.convert_to_tensor(t_pde, dtype=tf.float32),
        tf.convert_to_tensor(x_ic, dtype=tf.float32),
        tf.convert_to_tensor(t_ic, dtype=tf.float32),
        tf.convert_to_tensor(C_ic, dtype=tf.float32),
        tf.convert_to_tensor(x_bc, dtype=tf.float32),
        tf.convert_to_tensor(t_bc, dtype=tf.float32),
        tf.convert_to_tensor(C_bc, dtype=tf.float32),
    )

# Training the model
def train_model():
    model = PINN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    x_pde, t_pde, x_ic, t_ic, C_ic, x_bc, t_bc, C_bc = generate_training_data()
    loss_history = []

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = loss_function(model, x_pde, t_pde, x_ic, t_ic, C_ic, x_bc, t_bc, C_bc)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        loss = train_step()
        loss_history.append(loss.numpy())
        if (epoch + 1) % 500 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.numpy():.4f}')

    # Save model weights
    model.save_weights("pinn_advection_diffusion.weights.h5")

    # Plot loss history
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

    return model

# Generate data for visualization
def generate_plot_data(model, Nx=100, Nt=50):
    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(t_min, t_max, Nt)
    X, T = np.meshgrid(x, t)
    xt = np.stack([X.flatten(), T.flatten()], axis=1)
    C_pred = model(tf.convert_to_tensor(xt, dtype=tf.float32)).numpy().reshape(Nt, Nx)
    return x, t, C_pred

# Create 3D animation
def create_animation(x, t, C_pred):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(x, t)

    # Set up color map
    norm = plt.Normalize(vmin=np.min(C_pred), vmax=np.max(C_pred))
    colors = cm.viridis(norm(C_pred))

    # Precompute surface for performance
    surf = [ax.plot_surface(X, T, C_pred, facecolors=colors, linewidth=0, antialiased=False)]

    def update(frame):
        ax.clear()
        Z = np.zeros_like(X)
        Z[frame, :] = C_pred[frame, :]

        color_frame = cm.viridis(norm(C_pred[frame, :]))
        color_frame = np.tile(color_frame, (X.shape[0], 1, 1))

        ax.plot_surface(X, T, Z, facecolors=color_frame, linewidth=0, antialiased=False)

        ax.set_xlabel('Position (m)', fontsize=12)
        ax.set_ylabel('Time (s)', fontsize=12)
        ax.set_zlabel('Concentration', fontsize=12)
        ax.set_title(f'Pollution Dispersion at t={t[frame]:.2f} s', fontsize=14)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(t_min, t_max)
        ax.set_zlim(0, 1)

        ax.view_init(elev=30, azim=45 + 0.5 * frame)  # Slowly rotate the view

    ani = FuncAnimation(fig, update, frames=len(t), interval=100, blit=False)

    # Add color bar
    mappable = cm.ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Concentration')

    # Save animation as a GIF
    ani.save(r"E:\Prakhar Assignments\4th sem grind\PDE\advection_simulation.gif", writer='pillow', fps=10)
    plt.show()

# Main execution
if __name__ == '__main__':
    model = train_model()
    x, t, C_pred = generate_plot_data(model)
    create_animation(x, t, C_pred) 
