import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import sys


# custom layer for normalization
class NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, length_scale, time_scale, **kwargs):
        super().__init__(**kwargs)
        self.length_scale = length_scale
        self.time_scale = time_scale

    def call(self, x):
        spatial = x[:, :2]
        temporal = x[:, 2:]
        spatial_normalized = spatial / self.length_scale
        temporal_normalized = temporal / self.time_scale
        return tf.concat([spatial_normalized, temporal_normalized], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"length_scale": self.length_scale, "time_scale": self.time_scale}
        )
        return config


class CraterDiffusionPINN(tf.keras.Model):
    '''
    Physics-Informed Neural Network (PINN) for simulating crater diffusion.
    The PINN is trained to solve the diffusion equation for a crater profile.
        Diffusion equation: dh/dt = kappa * (d^2h/dx^2 + d^2h/dy^2)
    The PINN is trained with a combination of initial, boundary, and physics loss terms.
    A learning rate schedule and mixed precision optimization are employed for training.
    
    Parameters:
    kappa: float
        Thermal diffusivity (m^2/s)
    radius: float
        Crater radius (m)
    r_max: float
        Maximum radial distance for sampling points (m)
    t_max: float
        Maximum time for sampling points (s)
    
    Attributes:
    kappa: float
        Thermal diffusivity (m^2/s)
    radius: float
        Crater radius (m)
    r_max: float
        Maximum radial distance for sampling points (m)
    t_max: float
        Maximum time for sampling points (s)
    length_scale: float
        Normalization scale for spatial coordinates
    time_scale: float
        Normalization scale for time coordinate
    norm_layer: NormalizationLayer
        Custom normalization layer for spatial and temporal coordinates
    learning_rate_schedule: tf.keras.optimizers.schedules.ExponentialDecay
        Learning rate schedule for the optimizer
    optimizer: tf.keras.mixed_precision.LossScaleOptimizer
        Optimizer with mixed precision and learning rate schedule
    model: tf.keras.Model
        Neural network model for the PINN

    Methods:
    _build_network()
        Build the neural network architecture with wider layers
    h_of_r_fasset2014_tf(r, D)
        TensorFlow implementation of the normalized crater depth h(r)/D
    sample_collocation_points(n_points)
        Generate collocation points considering diffusion length scale
    visualize_collocation_points(n_points=1000)
        Helper function to visualize the sampling distribution
    sample_initial_points(n_points)
        Generate points for initial condition
    sample_boundary_points(n_points)
        Generate points on the spatial boundaries at different times
    compute_initial_condition_loss(x, y, t)
        Compute loss for initial condition
    compute_boundary_condition_loss(x, y, t)
        Compute loss for boundary conditions (zero at boundaries) with regularization
    compute_physics_loss(x, y, t)
        Compute the PDE residual loss using normalized variables
    train_step(n_points=1000)
        Single training step
    train(n_epochs=10000, n_points=1000)
        Train the PINN with adjusted weights and gradient clipping
    predict(x, y, t)
        Predict height at given points
        '''
    def __init__(self, kappa, radius, r_max, t_max):
        super().__init__()

        self.kappa = kappa
        self.radius = radius
        self.r_max = r_max
        self.t_max = t_max

        # Normalization scales
        self.length_scale = 2 * self.radius
        self.time_scale = self.length_scale**2 / self.kappa

        # Update normalization layer
        self.norm_layer = NormalizationLayer(self.length_scale, self.time_scale)
        if hasattr(self, "model"):
            self.model.layers[1].length_scale = self.length_scale
            self.model.layers[1].time_scale = self.time_scale

        # Optimizer with learning rate schedule
        initial_learning_rate = 0.01
        decay_steps = 1000
        decay_rate = 0.85
        self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps, decay_rate
        )

        # # Loss scale optimizer
        inner_optimizer = self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_schedule, clipnorm=1.0
        )
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            inner_optimizer, initial_scale=2.0**15
        )

        # Build the network
        # self.model = self._build_network()

    def _build_network(self):
        # """Build the neural network architecture with wider layers"""
        # inputs = tf.keras.layers.Input(shape=(3,))  # (x, y, t)

        # # Use the custom normalization layer
        # x = self.norm_layer(inputs)
        # x = tf.keras.layers.BatchNormalization()(x)

        # # Wider initial layer with dropout
        # h = tf.keras.layers.Dense(128, activation='swish')(x)
        # h = tf.keras.layers.Dropout(0.1)(h)

        # # More residual blocks with wider layers
        # for _ in range(4):  # Increased number of blocks
        #     h_new = tf.keras.layers.Dense(128, activation='swish')(h)
        #     h_new = tf.keras.layers.BatchNormalization()(h_new)
        #     h = tf.keras.layers.Add()([h, h_new])

        # # Additional dense layer before output
        # h = tf.keras.layers.Dense(64, activation='swish')(h)
        # outputs = tf.keras.layers.Dense(1)(h)

        """Build the neural network architecture"""
        inputs = tf.keras.layers.Input(shape=(3,))  # (x, y, t)

        # Use the custom normalization layer instead of Lambda
        x = self.norm_layer(inputs)
        x = tf.keras.layers.BatchNormalization()(x)

        # Initial layer with dropout
        h = tf.keras.layers.Dense(64, activation="swish")(x)
        h = tf.keras.layers.Dropout(0.1)(h)

        # Residual blocks
        for _ in range(4):
            h_new = tf.keras.layers.Dense(64, activation="swish")(h)
            h_new = tf.keras.layers.BatchNormalization()(h_new)
            h = tf.keras.layers.Add()([h, h_new])

        outputs = tf.keras.layers.Dense(1)(h)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def h_of_r_fasset2014_tf(self, r, D):
        """
        TensorFlow implementation of the normalized crater depth h(r)/D.

        Parameters:
        r: Tensor
            Radial distance from crater center (m)
        D: float
            Crater diameter (m)

        Returns:
        Tensor
            Normalized crater depth h(r)/D
        """
        R = D / 2  # Crater radius
        r = tf.abs(r)  # Ensure positive values

        # Central Flat Floor: r ≤ 0.2R
        mask_central = r <= 0.2 * R
        h_central = tf.where(mask_central, -0.181, 0.0)

        # Interior: 0.2R < r < 0.98R
        mask_interior = (r > 0.2 * R) & (r < 0.98 * R)
        h_interior = tf.where(
            mask_interior,
            -0.229 + 0.228 * (r / R) + 0.083 * (r / R) ** 2 - 0.039 * (r / R) ** 3,
            0.0,
        )

        # Rim and Exterior: 0.98R < r < 1.5R
        mask_rim = (r >= 0.98 * R) & (r < 1.5 * R)
        h_rim = tf.where(
            mask_rim,
            0.188 - 0.187 * (r / R) + 0.018 * (r / R) ** 2 + 0.015 * (r / R) ** 3,
            0.0,
        )

        # Combine all regions
        h_D = h_central + h_interior + h_rim

        return h_D

    def sample_collocation_points(self, n_points):
        """Generate collocation points considering diffusion length scale"""
        # Split points between different sampling strategies
        n_uniform = n_points // 2
        n_focused = n_points - n_uniform

        # # Sample times from a log-uniform distribution
        # time_exp = tf.random.uniform((n_points, 1))
        # t = self.t_max * tf.exp(tf.math.log(1e-2) * (1 - time_exp))

        t = self.t_max * tf.random.uniform((n_points, 1))

        # Sample spatial points with density proportional to 1/length_scale
        # For uniform points
        x_uniform = tf.random.uniform((n_uniform, 1), -self.r_max, self.r_max)
        y_uniform = tf.random.uniform((n_uniform, 1), -self.r_max, self.r_max)

        # For focused points near rim and floor
        theta = tf.random.uniform((n_focused, 1), 0, 2 * np.pi)
        r_focused = tf.concat(
            [
                self.radius
                * (0.08 + 0.9 * tf.random.uniform((n_focused // 2, 1))),  # Interior
                self.radius
                * (0.98 + 0.52 * tf.random.uniform((n_focused // 2, 1))),  # Rim
            ],
            axis=0,
        )
        # spread = length_scale[n_uniform:] # Use corresponding time points
        x_focused = r_focused * tf.cos(
            theta
        )  
        y_focused = r_focused * tf.sin(
            theta
        )  

        x = tf.concat([x_uniform, x_focused], axis=0)
        y = tf.concat([y_uniform, y_focused], axis=0)

        return x, y, t

    def visualize_collocation_points(self, n_points=1000):
        """Helper function to visualize the sampling distribution"""
        x, y, t = self.sample_collocation_points(n_points)

        plt.figure(figsize=(15, 5))

        # Spatial distribution
        plt.subplot(121)
        plt.scatter(x, y, c=t, alpha=0.5, s=1, cmap="gist_earth")
        circle = plt.Circle((0, 0), self.radius, fill=False, color="red")
        plt.gca().add_artist(circle)
        plt.colorbar(label="Time")
        plt.axis("equal")
        plt.title("Spatial Distribution of Collocation Points")

        # Temporal distribution
        plt.subplot(122)
        plt.hist(t.numpy(), bins=50)
        plt.title("Temporal Distribution of Collocation Points")
        plt.xlabel("Time")

        plt.tight_layout()
        plt.show()

    def sample_initial_points(self, n_points):
        """Generate points for initial condition"""

        # sample more densely near where the crater profile changes the most
        x, y, _ = self.sample_collocation_points(2 * n_points)
        t = tf.zeros_like(x)  # Set all times to 0

        return x, y, t

    def sample_boundary_points(self, n_points):
        """Generate points on the spatial boundaries at different times"""
        # Number of points per boundary
        n_per_boundary = n_points // 4

        # Time points for all boundaries
        t = tf.random.uniform((n_points, 1), 0, self.t_max)

        # Bottom boundary (y = -r_max)
        x_bottom = tf.random.uniform((n_per_boundary, 1), -self.r_max, self.r_max)
        y_bottom = tf.fill((n_per_boundary, 1), -self.r_max)

        # Top boundary (y = r_max)
        x_top = tf.random.uniform((n_per_boundary, 1), -self.r_max, self.r_max)
        y_top = tf.fill((n_per_boundary, 1), self.r_max)

        # Left boundary (x = -r_max)
        x_left = tf.fill((n_per_boundary, 1), -self.r_max)
        y_left = tf.random.uniform((n_per_boundary, 1), -self.r_max, self.r_max)

        # Right boundary (x = r_max)
        x_right = tf.fill((n_per_boundary, 1), self.r_max)
        y_right = tf.random.uniform((n_per_boundary, 1), -self.r_max, self.r_max)

        # Combine all boundaries
        x = tf.concat([x_bottom, x_top, x_left, x_right], axis=0)
        y = tf.concat([y_bottom, y_top, y_left, y_right], axis=0)

        return x, y, t

    @tf.function
    def compute_initial_condition_loss(self, x, y, t):
        """Compute loss for initial condition"""
        # Get predicted values at t=0
        points = tf.stack([x, y, t], axis=1)
        h_pred = self.model(points)

        # Calculate initial condition (fasset 2014 crater shape)
        r = tf.sqrt(x**2 + y**2)
        h_true = self.h_of_r_fasset2014_tf(r, 2 * self.radius)

        scale = 1e3
        h_diff = scale * (h_pred - h_true)
        ic_loss = tf.reduce_mean(tf.square(h_diff))

        return ic_loss

    @tf.function
    def compute_boundary_condition_loss(self, x, y, t):
        """Compute loss for boundary conditions (zero at boundaries) with regularization"""
        points = tf.stack([x, y, t], axis=1)
        h_pred = self.model(points)

        boundary_loss = tf.reduce_mean(tf.square(h_pred))

        return boundary_loss

    @tf.function
    def compute_physics_loss(self, x, y, t):
        """Compute the PDE residual loss using normalized variables"""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y, t])
                points = tf.stack([x, y, t], axis=1)
                h = self.model(points)

            dh_dx = tape1.gradient(h, x)
            dh_dy = tape1.gradient(h, y)
            dh_dt = tape1.gradient(h, t)

        d2h_dx2 = tape2.gradient(dh_dx, x)
        d2h_dy2 = tape2.gradient(dh_dy, y)

        del tape1, tape2  # Free memory

        self.pde_scale = 1e6  # for gradient stability, the derivatives are super small.
        pde_residual = self.pde_scale * (dh_dt - self.kappa * (d2h_dx2 + d2h_dy2))

        return tf.reduce_mean(tf.square(pde_residual))

    @tf.function
    def train_step(self, n_points=1000):
        """Single training step"""
        with tf.GradientTape() as tape:
            # Sample points
            x_pde, y_pde, t_pde = self.sample_collocation_points(n_points)
            x_ic, y_ic, t_ic = self.sample_initial_points(n_points)
            x_bc, y_bc, t_bc = self.sample_boundary_points(n_points)

            # Compute losses
            pde_loss = self.compute_physics_loss(x_pde, y_pde, t_pde)
            ic_loss = self.compute_initial_condition_loss(x_ic, y_ic, t_ic)
            bc_loss = self.compute_boundary_condition_loss(x_bc, y_bc, t_bc)

            # # Add L2 regularization to the loss
            # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
            # l2m = 1e-4 # Adjust the regularization factor as needed

            # Total loss (weighted sum)
            total_loss = pde_loss + ic_loss + bc_loss  # + l2m * l2_loss

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return total_loss, pde_loss, ic_loss, bc_loss

    def train(self, n_epochs=10000, n_points=1000):
        """Train the PINN with adjusted weights and gradient clipping"""

        # Keep full history for plotting
        all_losses = []

        # Setup progress bar
        pbar = tqdm(range(n_epochs))

        patience = 5000  # Number of epochs to wait for improvement
        min_delta = 1e-5  # Minimum change to qualify as an improvement
        best_loss = float("inf")
        wait = 0

        for epoch in pbar:
            # Training step
            total_loss, pde_loss, ic_loss, bc_loss = self.train_step(n_points)

            # Store in both recent and full history
            all_losses.append(
                [float(total_loss), float(pde_loss), float(ic_loss), float(bc_loss)]
            )

            # Update progress bar (only compute string format if tqdm will display it)
            if epoch % 10 == 0:
                pbar.set_description(
                    f"Loss: {float(total_loss):.2e} (PDE: {float(pde_loss):.1e} IC: {float(ic_loss):.1e} BC: {float(bc_loss):.1e})"
                )

            # Early stopping check
            if total_loss < best_loss - min_delta:
                best_loss = total_loss
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 100 == 0:
                # update the network scaling parameters
                r_exp = np.random.uniform(size=1001)
                lower_r = 13
                upper_r = 1500 + lower_r
                y_ = np.array(
                    upper_r * tf.exp(tf.math.log(1e-2) * (1 - r_exp)) - lower_r
                )
                new_r = y_[
                    np.random.randint(0, 1000)
                ]  # sample from the exponential distribution of radius values

                t_max_new = np.min((1.5 * np.ceil(new_r**2 / self.kappa), 4500))

                tf.print(f"r0: {new_r:.2f}, t_max: {t_max_new:.2f}", end="\r")

                self.__init__(self.kappa, new_r, 4.0 * new_r, t_max_new)

        return np.array(all_losses)

    def predict(self, x, y, t):
        """Predict height at given points"""
        points = tf.stack([x, y, t], axis=1)

        return self.model(points).numpy()
