"""
Model architecture of Mirage model
"""

import os

from utils import lazy_scope

try:
    import tensorflow.compat.v1 as tf

    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

import tensorflow_probability as tfp

tf.set_random_seed(24)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Mirage:
    """Class for the Mirage model"""

    def __init__(
        self,
        input_size,
        latent_dim=10,
        som_dim=[8, 8],
        learning_rate=1e-4,
        decay_factor=0.99,
        decay_steps=2000,
        input_channels=98,
        alpha=10.0,
        beta=100.0,
        gamma=100.0,
        kappa=0.0,
        tau=1.0,
        theta=1.0,
        eta=1.0,
        dropout=0.5,
        prior=0.001,
        conditional_loss_weight=1000,
        lstm_dim=100,
        condition_size=3,
        trans_mat_size=5,
        trans_initial_bias=[],
        trans_class_weights=[],
    ):
        """Initialization method for the Mirage model object.
        Args:
            input_size (int): Length of the input vector.
            latent_dim (int): The dimensionality of the latent embeddings (default: 100).
            som_dim (list): The dimensionality of the self-organizing map (default: [8,8]).
            learning_rate (float): The learning rate for the optimization (default: 1e-4).
            decay_factor (float): The factor for the learning rate decay (default: 0.99).
            decay_steps (int): The number of optimization steps before every learning rate
                decay (default: 1000).
            input_channels (int): The number of channels of the input data points (default: 98).
            alpha (float): The weight for the commitment loss (default: 10.).
            beta (float): Weight for the SOM loss (default: 100).
            gamma (float): Weight for the KL term of the PSOM clustering loss (default: 100).
            kappa (float): Weight for the smoothness loss (default: 10).
            theta (float): Weight for the VAE loss (default: 1).
            eta (float): Weight for the prediction loss (default: 1).
            dropout (float): Dropout factor for the feed-forward layers of the VAE (default: 0.5).
            prior (float): Weight of the regularization term of the ELBO (default: 0.5).
        """

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.som_dim = som_dim
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.input_channels = input_channels
        self.condition_size = condition_size
        self.trans_mat_size = trans_mat_size
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.theta = theta
        self.kappa = kappa
        self.tau = tau
        self.dropout = dropout
        self.prior = prior
        self.conditional_loss_weight = conditional_loss_weight
        self.lstm_dim = lstm_dim
        self.init_bias = trans_initial_bias
        self.class_weights = trans_class_weights
        self.prior
        self.conditional_loss_weight
        self.is_training
        self.inputs
        self.x_next_inputs
        self.concatenated_inputs
        self.x_trans_mat_inputs
        self.y_trans_mat_inputs
        self.conditions
        self.x
        self.x_next
        self.concat_x_c
        self.x_trans_mat
        self.y_trans_mat
        self.batch_size
        self.step_size
        self.embeddings
        self.global_step
        self.shift_x_trans_mat
        self.condition_pred_y
        self.z_e
        self.z_e_sample
        self.k
        self.prediction
        self.next_z_e_identity
        self.z_e_old
        self.z_dist_flat
        self.z_q
        self.z_q_neighbors
        self.reconstruction_e
        self.reconstruction_e_identity
        self.loss_reconstruction_ze
        self.q
        self.p
        self.loss_commit
        self.loss_som
        self.loss_prediction
        self.loss_smoothness
        self.loss
        self.loss_a
        self.loss_shift
        self.loss_conditional
        self.loss_transition
        self.optimize
        self.loss_forecasting
        self.damp_factor
        self.damp_prediction

    @lazy_scope
    def is_training(self):
        is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
        return is_training

    @lazy_scope
    def x_trans_mat_inputs(self):
        x_trans_mat = tf.placeholder(
            tf.float32, shape=[None, None, self.trans_mat_size], name="x_trans_mat"
        )
        return x_trans_mat

    @lazy_scope
    def x_trans_mat(self):
        x_trans_mat = tf.reshape(self.x_trans_mat_inputs, [-1, self.trans_mat_size])
        return x_trans_mat

    @lazy_scope
    def y_trans_mat_inputs(self):
        y_trans_mat = tf.placeholder(
            tf.float32, shape=[None, None, self.trans_mat_size], name="y_trans_mat"
        )
        return y_trans_mat

    @lazy_scope
    def y_trans_mat(self):
        y_trans_mat = tf.reshape(self.y_trans_mat_inputs, [-1, self.trans_mat_size])
        return y_trans_mat

    @lazy_scope
    def inputs(self):
        x = tf.placeholder(
            tf.float32, shape=[None, None, self.input_channels], name="x"
        )
        return x

    @lazy_scope
    def x_next_inputs(self):
        x_next = tf.placeholder(
            tf.float32, shape=[None, None, self.input_channels], name="x_next"
        )
        return x_next

    @lazy_scope
    def conditions(self):
        c = tf.placeholder(
            tf.float32, shape=[None, None, self.condition_size], name="c"
        )
        return c

    @lazy_scope
    def concatenated_inputs(self):
        concat_x_c = tf.placeholder(
            tf.float32,
            shape=[None, None, self.input_channels + self.condition_size],
            name="concat_x_c",
        )
        return concat_x_c

    @lazy_scope
    def x(self):
        x = tf.reshape(self.inputs, [-1, self.input_channels])
        return x

    @lazy_scope
    def x_next(self):
        x_next = tf.reshape(self.x_next_inputs, [-1, self.input_channels])
        return x_next

    @lazy_scope
    def c(self):
        c = tf.reshape(self.conditions, [-1, self.condition_size])
        return c

    @lazy_scope
    def concat_x_c(self):
        concat_x_c = tf.reshape(
            self.concatenated_inputs, [-1, self.input_channels + self.condition_size]
        )
        return concat_x_c

    @lazy_scope
    def batch_size(self):
        """Reads the batch size from the input tensor."""
        batch_size = tf.shape(self.inputs)[0]
        return batch_size

    @lazy_scope
    def step_size(self):
        """Reads the step size from the input tensor."""
        step_size = tf.shape(self.inputs)[1]
        return step_size

    @lazy_scope
    def embeddings(self):
        """Creates variable for the SOM embeddings."""
        embeddings = tf.get_variable(
            "embeddings",
            self.som_dim + [self.latent_dim],
            initializer=tf.truncated_normal_initializer(stddev=0.05),
        )
        tf.summary.tensor_summary("embeddings", embeddings)
        return embeddings

    @lazy_scope
    def global_step(self):
        """Creates global_step variable for the optimization."""
        global_step = tf.Variable(0, trainable=False, name="global_step")
        return global_step

    @lazy_scope
    def k1_input(self):
        # Placeholder for k1 of som - som_dim[0]
        return tf.placeholder(tf.int32)

    @lazy_scope
    def k2_input(self):
        # Placeholder for k2 of som - som_dim[1]
        return tf.placeholder(tf.int32)

    @lazy_scope
    def pred_k_input(self):
        # Placeholder for k1 of som - som_dim[0]
        return tf.placeholder(tf.int32, shape=(self.batch_size * self.step_size,))

    @lazy_scope
    def current_k_input(self):
        # Placeholder for k2 of som - som_dim[1]
        return tf.placeholder(tf.int32, shape=(self.batch_size * self.step_size,))

    @lazy_scope
    def z_e(self):
        """Computes the distribution of probability of the latent embeddings."""
        with tf.variable_scope("encoder"):
            cond_tf = self.condition_y(self.shift_x_trans_mat)
            cond_one_hot = tf.one_hot(
                tf.argmax(cond_tf, dimension=1), depth=self.condition_size
            )
            concat_x_c_ip = tf.concat([self.x, cond_one_hot], -1)

            h_1 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(concat_x_c_ip)

            h_1 = tf.keras.layers.Dropout(rate=self.dropout)(h_1)
            h_1 = tf.keras.layers.BatchNormalization()(h_1)
            h_1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(h_1)
            h_1 = tf.keras.layers.Dropout(rate=self.dropout)(h_1)
            h_1 = tf.keras.layers.BatchNormalization()(h_1)
            h_2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(h_1)
            h_2 = tf.keras.layers.Dropout(rate=self.dropout)(h_2)
            h_2 = tf.keras.layers.BatchNormalization()(h_2)
            z_e_mu = tf.keras.layers.Dense(self.latent_dim, activation=None)(h_2)
            z_e_sigma = tf.keras.layers.Dense(self.latent_dim, activation=None)(h_2)
            z_e = tfp.distributions.MultivariateNormalDiag(
                loc=z_e_mu, scale_diag=tfp.bijectors.Softplus()(z_e_sigma)
            )
        return z_e

    @lazy_scope
    def next_z_e_identity(self):
        """Sample from the distribution of probability of the next latent embeddings genereated by prediction"""
        next_z_e = tf.identity(self.prediction, name="next_z_e")
        print("next_z_e Identity shape:", next_z_e.shape)
        return next_z_e

    @lazy_scope
    def reconstruction_e_identity(self):
        """Sample from the distribution of probability of the latent embeddings."""
        x_recons = tf.identity(self.reconstruction_e, name="x_recons")
        print("x_recons Identity shape:", x_recons.shape)
        return x_recons

    @lazy_scope
    def shift_x_trans_mat_identity(self):
        """Sample from the distribution of probability of the latent embeddings."""
        shift_x_trans_mat_iden = tf.identity(
            self.shift_x_trans_mat, name="shift_x_trans_mat_iden"
        )
        print("shift_x_trans_mat_iden Identity shape:", shift_x_trans_mat_iden.shape)
        return shift_x_trans_mat_iden

    @lazy_scope
    def z_e_sample(self):
        """Sample from the distribution of probability of the latent embeddings."""
        z_e = self.z_e.sample()
        print("Self.z_e.sample shape:", z_e.shape)

        z_e = tf.identity(z_e, name="z_e")
        print("z_e Identity shape:", z_e.shape)

        tf.summary.histogram("count_nonzeros_z_e", tf.count_nonzero(z_e, -1))
        return z_e

    @lazy_scope
    def z_e_old(self):
        """Aggregates the encodings of the respective previous time steps."""
        z_e_old = tf.concat([self.z_e_sample[0:1], self.z_e_sample[:-1]], axis=0)
        return z_e_old

    @lazy_scope
    def z_dist_flat(self):
        """Computes the distances between the centroids and the embeddings."""
        z_dist = tf.squared_difference(
            tf.expand_dims(tf.expand_dims(self.z_e_sample, 1), 1),
            tf.expand_dims(self.embeddings, 0),
        )
        print("z_dist shape:", z_dist.shape)

        z_dist_red = tf.reduce_sum(z_dist, axis=-1)
        print("z_dist_red shape:", z_dist_red.shape)

        z_dist_flat = tf.reshape(z_dist_red, [-1, self.som_dim[0] * self.som_dim[0]])
        print("z_dist_flat shape:", z_dist_flat.shape)

        return z_dist_flat

    @lazy_scope
    def z_dist_flat_ng(self):
        """Computes the distances between the centroids and the embeddings stopping the gradient of the latent
        embeddings."""
        z_dist = tf.squared_difference(
            tf.expand_dims(tf.expand_dims(tf.stop_gradient(self.z_e_sample), 1), 1),
            tf.expand_dims(self.embeddings, 0),
        )
        z_dist_red = tf.reduce_sum(z_dist, axis=-1)
        z_dist_flat = tf.reshape(z_dist_red, [-1, self.som_dim[0] * self.som_dim[1]])
        return z_dist_flat

    @lazy_scope
    def k(self):
        """Picks the index of the closest centroid for every embedding."""
        k = tf.argmin(self.z_dist_flat, axis=-1, name="k")
        tf.summary.histogram("clusters", k)
        return k

    @lazy_scope
    def pred_k(self):
        """Picks the index of the closest centroid for predicted embedding."""
        predicted_z_e_sample = self.prediction.sample()
        predicted_z_dist = tf.squared_difference(
            tf.expand_dims(tf.expand_dims(predicted_z_e_sample, 1), 1),
            tf.expand_dims(self.embeddings, 0),
        )
        predicted_z_dist_red = tf.reduce_sum(predicted_z_dist, axis=-1)
        print("predicted_z_dist_red shape:", predicted_z_dist_red.shape)
        predicted_z_dist_flat = tf.reshape(
            predicted_z_dist_red, [-1, self.som_dim[0] * self.som_dim[0]]
        )
        print("predicted_z_dist_flat shape:", predicted_z_dist_flat.shape)
        predicted_k = tf.argmin(predicted_z_dist_flat, axis=-1, name="pred_k")
        tf.summary.histogram("pred_clusters", predicted_k)
        return predicted_k

    @lazy_scope
    def prediction(self):
        """Predict the distribution of probability of the next embedding."""
        with tf.variable_scope("next_state"):
            z_e_p = tf.placeholder(
                tf.float32, shape=[None, None, self.latent_dim], name="input_lstm"
            )
            z_e = tf.cond(self.is_training, lambda: self.z_e_sample, lambda: z_e_p)

            rnn_input = tf.stop_gradient(
                tf.reshape(z_e, [self.batch_size, self.step_size, self.latent_dim])
            )

            init_state_p = tf.placeholder(
                tf.float32, shape=[2, None, self.lstm_dim], name="init_state"
            )

            cell = tf.keras.layers.LSTM(
                self.lstm_dim, return_sequences=True, return_state=True
            )
            init_state = cell.get_initial_state(rnn_input)
            state = tf.cond(
                self.is_training,
                lambda: init_state,
                lambda: [init_state_p[0], init_state_p[1]],
            )
            lstm_output, state_h, state_c = cell(rnn_input, initial_state=state)
            state = tf.identity([state_h, state_c], name="next_state")
            print("lstm_output shape", lstm_output.shape)

            # Calculate attention scores
            attention_scores = tf.keras.layers.Dot(axes=2)([lstm_output, lstm_output])
            print("attention_scores shape", attention_scores.shape)

            # Apply softmax to the attention scores
            attention_weights = tf.keras.layers.Softmax()(attention_scores)
            print("attention_weights shape", attention_weights.shape)

            # Apply identity function to fetch it later.
            attention_weights_identity = tf.identity(
                attention_weights, name="attention_weights"
            )

            # Apply attention weights to lstm_output
            context_vector = tf.keras.layers.Dot(axes=[2, 1])(
                [attention_weights, lstm_output]
            )
            print("context_vector shape", context_vector.shape)

            # Concatenate context_vector with lstm_output
            lstm_output_with_attention = tf.keras.layers.Concatenate(axis=-1)(
                [lstm_output, context_vector]
            )
            print("lstm_output_with_attention shape", lstm_output_with_attention.shape)

            # Reshape lstm_output_with_attention
            lstm_output_with_attention = tf.reshape(
                lstm_output_with_attention,
                [self.batch_size * self.step_size, 2 * self.lstm_dim],
            )
            print("lstm_output_with_attention shape", lstm_output_with_attention.shape)

            # Apply a dense layer with activation
            h_1 = tf.keras.layers.Dense(self.lstm_dim, activation=tf.nn.leaky_relu)(
                lstm_output_with_attention
            )

            # Apply another dense layer to predict the next_z_e and IndependentNormal layer
            next_z_e = tf.keras.layers.Dense(
                tfp.layers.IndependentNormal.params_size(self.latent_dim),
                activation=None,
            )(h_1)
            next_z_e = tfp.layers.IndependentNormal(self.latent_dim)(next_z_e)

        next_z_e_sample = tf.reshape(
            tf.identity(next_z_e),
            [-1, self.step_size, self.latent_dim],
            name="next_z_e",
        )

        return next_z_e

    @lazy_scope
    def z_q(self):
        """Aggregates the respective closest embedding for every centroid."""
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        k_stacked = tf.stack([k_1, k_2], axis=1)
        z_q = tf.gather_nd(self.embeddings, k_stacked, name="z_q")
        return z_q

    @lazy_scope
    def z_q_neighbors(self):
        """Aggregates the respective neighbors in the SOM grid for every z_q."""
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        k_stacked = tf.stack([k_1, k_2], axis=1)

        k1_not_top = tf.less(k_1, tf.constant(self.som_dim[0] - 1, dtype=tf.int64))
        k1_not_bottom = tf.greater(k_1, tf.constant(0, dtype=tf.int64))
        k2_not_right = tf.less(k_2, tf.constant(self.som_dim[1] - 1, dtype=tf.int64))
        k2_not_left = tf.greater(k_2, tf.constant(0, dtype=tf.int64))

        k1_up = tf.where(
            k1_not_top, tf.add(k_1, 1), tf.zeros(tf.shape(k_1), dtype=tf.dtypes.int64)
        )
        k1_down = tf.where(
            k1_not_bottom,
            tf.subtract(k_1, 1),
            tf.ones(tf.shape(k_1), dtype=tf.dtypes.int64) * (self.som_dim[0] - 1),
        )
        k2_right = tf.where(
            k2_not_right, tf.add(k_2, 1), tf.zeros(tf.shape(k_2), dtype=tf.dtypes.int64)
        )
        k2_left = tf.where(
            k2_not_left,
            tf.subtract(k_2, 1),
            tf.ones(tf.shape(k_2), dtype=tf.dtypes.int64) * (self.som_dim[0] - 1),
        )

        z_q_up = tf.gather_nd(self.embeddings, tf.stack([k1_up, k_2], axis=1))
        z_q_down = tf.gather_nd(self.embeddings, tf.stack([k1_down, k_2], axis=1))
        z_q_right = tf.gather_nd(self.embeddings, tf.stack([k_1, k2_right], axis=1))
        z_q_left = tf.gather_nd(self.embeddings, tf.stack([k_1, k2_left], axis=1))

        z_q_neighbors = tf.stack(
            [self.z_q, z_q_up, z_q_down, z_q_right, z_q_left], axis=1
        )
        return z_q_neighbors

    def call_decoder(self, input_z_e_concat_trans):
        """Reconstructs the input from the encodings by learning a Gaussian distribution."""
        with tf.variable_scope("decoder_arch", reuse=tf.AUTO_REUSE):
            h_1 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)(
                input_z_e_concat_trans
            )
            h_1 = tf.keras.layers.BatchNormalization()(h_1)
            h_2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(h_1)
            h_2 = tf.keras.layers.BatchNormalization()(h_2)
            h_3 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(h_2)
            h_3 = tf.keras.layers.BatchNormalization()(h_3)
            x_hat = tf.keras.layers.Dense(self.input_channels, activation=None)(h_3)
        x_hat = tf.identity(x_hat, name="x_hat")
        return x_hat

    @lazy_scope
    def reconstruction_e(self):
        """Reconstructs the input from the encodings by learning a Gaussian distribution."""
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            z_p = tf.placeholder(
                tf.float32,
                shape=[None, self.latent_dim + self.condition_size],
                name="z_e",
            )

            cond_tf = self.condition_y(tf.stop_gradient(self.shift_x_trans_mat))
            cond_one_hot = tf.one_hot(
                tf.argmax(cond_tf, dimension=1), depth=self.condition_size
            )
            z_e_concat_trans = tf.concat([self.z_e_sample, cond_one_hot], -1)
            z_e = tf.cond(self.is_training, lambda: z_e_concat_trans, lambda: z_p)

        return self.call_decoder(z_e)

    @lazy_scope
    def loss_reconstruction_ze(self):
        """Computes the ELBO."""
        prior = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self.latent_dim), scale_diag=tf.ones(self.latent_dim)
        )
        kl_loss = tf.reduce_mean(self.z_e.kl_divergence(prior))

        log_lik_loss = tf.reduce_mean(
            tf.squared_difference(self.reconstruction_e, self.x)
        )

        print("KL loss:", kl_loss)
        print("Log Likelihood loss:", log_lik_loss)

        loss_rec_mse_ze = self.prior * kl_loss + log_lik_loss
        tf.summary.scalar("log_lik_loss", log_lik_loss)
        tf.summary.scalar("kl_loss", kl_loss)
        tf.summary.scalar("loss_reconstruction_ze", loss_rec_mse_ze)
        return loss_rec_mse_ze

    @lazy_scope
    def q(self):
        """Computes the soft assignments between the embeddings and the centroids."""
        with tf.name_scope("distribution"):
            q = tf.keras.backend.epsilon() + 1.0 / (
                1.0 + self.z_dist_flat / self.alpha
            ) ** ((self.alpha + 1.0) / 2.0)
            q = q / tf.reduce_sum(q, axis=1, keepdims=True)
            q = tf.identity(q, name="q")
            print("q shape", q.shape)
        return q

    @lazy_scope
    def q_ng(self):
        """Computes the soft assignments between the embeddings and the centroids stopping the gradient of the latent
        embeddings."""
        with tf.name_scope("distribution"):
            q = tf.keras.backend.epsilon() + 1.0 / (
                1.0 + self.z_dist_flat_ng / self.alpha
            ) ** ((self.alpha + 1.0) / 2.0)
            q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        return q

    @lazy_scope
    def p(self):
        """Placeholder for the target distribution."""
        p = tf.placeholder(tf.float32, shape=(None, self.som_dim[0] * self.som_dim[1]))
        return p

    def target_distribution(self, q):
        """Computes the target distribution given the soft assignment between embeddings and centroids."""
        p = q**2 / (q.sum(axis=0))
        p = p / p.sum(axis=1, keepdims=True)
        return p

    @lazy_scope
    def loss_som(self):
        """Computes the SOM loss."""
        k = tf.range(self.som_dim[0] * self.som_dim[1])
        k_1 = k // self.som_dim[0]
        k_2 = k % self.som_dim[1]

        k1_not_top = tf.less(k_1, tf.constant(self.som_dim[0] - 1, dtype=tf.int32))
        k1_not_bottom = tf.greater(k_1, tf.constant(0, dtype=tf.int32))
        k2_not_right = tf.less(k_2, tf.constant(self.som_dim[1] - 1, dtype=tf.int32))
        k2_not_left = tf.greater(k_2, tf.constant(0, dtype=tf.int32))

        k1_up = tf.where(
            k1_not_top, tf.add(k_1, 1), tf.zeros(tf.shape(k_1), dtype=tf.dtypes.int32)
        )
        k1_down = tf.where(
            k1_not_bottom,
            tf.subtract(k_1, 1),
            tf.ones(tf.shape(k_1), dtype=tf.dtypes.int32) * (self.som_dim[0] - 1),
        )
        k2_right = tf.where(
            k2_not_right, tf.add(k_2, 1), tf.zeros(tf.shape(k_2), dtype=tf.dtypes.int32)
        )
        k2_left = tf.where(
            k2_not_left,
            tf.subtract(k_2, 1),
            tf.ones(tf.shape(k_2), dtype=tf.dtypes.int32) * (self.som_dim[0] - 1),
        )

        k_up = k1_up * self.som_dim[0] + k_2
        k_down = k1_down * self.som_dim[0] + k_2
        k_right = k_1 * self.som_dim[0] + k2_right
        k_left = k_1 * self.som_dim[0] + k2_left

        q_t = tf.transpose(self.q_ng)
        q_up = tf.transpose(
            tf.gather_nd(q_t, tf.reshape(k_up, [self.som_dim[0] * self.som_dim[1], 1]))
        )
        q_down = tf.transpose(
            tf.gather_nd(
                q_t, tf.reshape(k_down, [self.som_dim[0] * self.som_dim[1], 1])
            )
        )
        q_right = tf.transpose(
            tf.gather_nd(
                q_t, tf.reshape(k_right, [self.som_dim[0] * self.som_dim[1], 1])
            )
        )
        q_left = tf.transpose(
            tf.gather_nd(
                q_t, tf.reshape(k_left, [self.som_dim[0] * self.som_dim[1], 1])
            )
        )

        q_neighbours = tf.concat(
            [
                tf.expand_dims(q_up, -1),
                tf.expand_dims(q_down, -1),
                tf.expand_dims(q_right, -1),
                tf.expand_dims(q_left, -1),
            ],
            axis=2,
        )
        q_neighbours = tf.reduce_sum(tf.math.log(q_neighbours), axis=-1)

        new_q = self.q
        q_n = tf.math.multiply(q_neighbours, tf.stop_gradient(new_q))
        q_n = tf.reduce_sum(q_n, axis=-1)
        qq = tf.math.negative(tf.reduce_mean(q_n))
        return qq

    def damp_nn(self, input_cond):
        with tf.variable_scope("damp_nn", reuse=tf.AUTO_REUSE):
            h1 = tf.keras.layers.Dense(100, activation=tf.nn.leaky_relu)(input_cond)
            h1 = tf.keras.layers.BatchNormalization()(h1)
            h1 = tf.keras.layers.Dense(10, activation=tf.nn.leaky_relu)(h1)
            h1 = tf.keras.layers.BatchNormalization()(h1)
            damp_factor = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(h1)
        return damp_factor

    @lazy_scope
    def damp_factor(self):
        logits_y_hat = self.condition_y(tf.stop_gradient(self.x_trans_mat))
        c = tf.argmax(logits_y_hat, dimension=1)
        c = tf.reshape(c, [-1, self.step_size])

        second_last_element = c[:, -2]
        last_element = c[:, -1]
        print("second_last_element.shape:", second_last_element.shape)
        print("last_element.shape:", last_element.shape)
        print("self.x_trans_mat.shape:", self.x_trans_mat.shape)
        trans_mat_tensor = tf.concat([self.x_trans_mat, self.shift_x_trans_mat], axis=1)
        print("trans_mat_tensor.shape:", trans_mat_tensor.shape)

        return self.damp_nn(trans_mat_tensor)

    @lazy_scope
    def damp_prediction(self):
        """Compute the damped prediction loss"""
        z_e = tf.reshape(
            self.z_e_sample, [self.batch_size, self.step_size, self.latent_dim]
        )
        z_e_next = tf.concat(
            [z_e[:, 1:], tf.reshape(z_e[:, -1], [-1, 1, self.latent_dim])], axis=1
        )
        z_e_next = tf.stop_gradient(tf.reshape(z_e_next, [-1, self.latent_dim]))

        loss_prediction = -tf.reduce_mean(
            tf.math.multiply(self.prediction.log_prob(z_e_next), self.damp_factor)
        )
        return loss_prediction

    @lazy_scope
    def loss_prediction(self):
        """Compute the prediction loss"""
        z_e = tf.reshape(
            self.z_e_sample, [self.batch_size, self.step_size, self.latent_dim]
        )
        z_e_next = tf.concat(
            [z_e[:, 1:], tf.reshape(z_e[:, -1], [-1, 1, self.latent_dim])], axis=1
        )
        z_e_next = tf.stop_gradient(tf.reshape(z_e_next, [-1, self.latent_dim]))

        loss_prob = -tf.reduce_mean(self.prediction.log_prob(z_e_next))

        return loss_prob

    @lazy_scope
    def loss_forecasting(self):
        """Compute the forecasting loss of reconstructed_x from the decoder with original_x."""
        shift_x_trans_mat_iden = self.shift_x_trans_mat_identity
        next_cond_o = self.condition_y(shift_x_trans_mat_iden)
        next_z_e_o = self.next_z_e_identity
        print("next_cond_o shape:", next_cond_o.shape)
        print("next_z_e_o shape:", next_z_e_o.shape)
        decoder_input = tf.concat([next_z_e_o, next_cond_o], axis=1)
        x_hat = self.call_decoder(decoder_input)

        print("self.x_next shape", self.x_next.shape)
        print("x_hat shape", x_hat.shape)
        loss_forecast = tf.reduce_mean(
            tf.reduce_sum(tf.squared_difference(self.x_next, x_hat), axis=-1), axis=-1
        )
        return loss_forecast

    @lazy_scope
    def loss_smoothness(self):
        """Compute the smoothness loss"""

        k_reshaped = tf.reshape(self.k, [self.batch_size, self.step_size])
        k_old = tf.concat([k_reshaped[:, 0:1], k_reshaped[:, :-1]], axis=1)
        k_old = tf.reshape(tf.cast(k_old, tf.int64), [-1, 1])
        emb = tf.reshape(
            self.embeddings, [self.som_dim[0] * self.som_dim[1], self.latent_dim]
        )
        e = tf.gather_nd(emb, k_old)
        diff = tf.reduce_sum(
            tf.squared_difference(self.z_e_sample, tf.stop_gradient(e)), axis=-1
        )
        q = tf.keras.backend.epsilon() + (
            1.0 / (1.0 + diff / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        )
        loss_smoothness = -tf.reduce_mean(q)
        return loss_smoothness

    @lazy_scope
    def loss(self):
        """Aggregates the loss terms into the total loss."""
        loss = (
            self.theta * self.loss_reconstruction_ze
            + self.beta * self.loss_som
            + self.kappa * self.loss_smoothness
            + self.gamma * self.loss_commit
            + self.tau * self.loss_transition
            + self.eta * self.damp_prediction
            + self.loss_forecasting
        )

        tf.summary.scalar("loss_rec", self.theta * self.loss_reconstruction_ze)
        tf.summary.scalar("loss_commit", self.gamma * self.loss_commit)
        tf.summary.scalar("loss_som", self.beta * self.loss_som)
        tf.summary.scalar("loss_smoothness", self.kappa * self.loss_smoothness)
        tf.summary.scalar("loss_damp_prediction", self.eta * self.damp_prediction)
        tf.summary.scalar("loss_trans", self.tau * self.loss_transition)
        tf.summary.scalar("loss", loss)

        return loss

    @lazy_scope
    def loss_commit(self):
        """Computes the KL term of the clustering loss."""
        loss_commit = tf.reduce_mean(
            tf.reduce_sum(
                self.p * tf.log((tf.keras.backend.epsilon() + self.p) / (self.q)),
                axis=1,
            )
        )

        return loss_commit

    @lazy_scope
    def loss_commit_sd(self):
        """Computes the commitment loss of standard SOM for initialization."""
        loss_commit = tf.reduce_mean(
            tf.squared_difference(tf.stop_gradient(self.z_e_sample), self.z_q)
        )
        tf.summary.scalar("loss_commit_sd", loss_commit)
        return loss_commit

    @lazy_scope
    def loss_som_old(self):
        """Computes the SOM loss."""
        loss_som = tf.reduce_mean(
            tf.squared_difference(
                tf.expand_dims(tf.stop_gradient(self.z_e_sample), axis=1),
                self.z_q_neighbors,
            )
        )
        tf.summary.scalar("loss_som_old", loss_som)
        return loss_som

    @lazy_scope
    def loss_a(self):
        """Aggregates the loss terms into the total loss."""
        loss = self.loss_som_old + self.loss_commit_sd
        tf.summary.scalar("loss_som_pre_total", loss)
        return loss

    @lazy_scope
    def optimize(self):
        """Optimizes the model's loss using Adam with exponential learning rate decay."""
        lr_decay = tf.train.exponential_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.decay_factor,
            staircase=True,
        )
        optimizer = tf.train.AdamOptimizer(lr_decay)

        train_step = optimizer.minimize(self.loss, global_step=self.global_step)
        train_step_prob = optimizer.minimize(
            self.damp_prediction, global_step=self.global_step
        )
        train_step_ae = optimizer.minimize(
            self.loss_reconstruction_ze, global_step=self.global_step
        )
        train_step_som = optimizer.minimize(self.loss_a, global_step=self.global_step)
        train_step_trans = optimizer.minimize(
            self.loss_transition, global_step=self.global_step
        )
        train_step_forecast = optimizer.minimize(
            self.loss_forecasting, global_step=self.global_step
        )

        return (
            train_step,
            train_step_ae,
            train_step_som,
            train_step_prob,
            train_step_trans,
            train_step_forecast,
        )

    @lazy_scope
    def shift_x_trans_mat(self):
        """Predicts the next shift - transition matrix for each player."""
        with tf.variable_scope("shift", reuse=tf.AUTO_REUSE):
            shift_h_1 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(
                self.x_trans_mat
            )
            shift_h_1 = tf.keras.layers.BatchNormalization()(shift_h_1)
            shift_h_1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(
                shift_h_1
            )
            shift_h_1 = tf.keras.layers.BatchNormalization()(shift_h_1)
            trans_mat_op = tf.keras.layers.Dense(
                self.trans_mat_size, activation=tf.nn.sigmoid
            )(shift_h_1)

        trans_mat_identity = tf.identity(trans_mat_op, name="trans_mat_op")
        return trans_mat_op

    def condition_y(self, input_y):
        with tf.variable_scope("cond_y", reuse=tf.AUTO_REUSE):
            cond_h_1 = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu)(input_y)
            cond_h_1 = tf.keras.layers.BatchNormalization()(cond_h_1)
            conditions_op = tf.keras.layers.Dense(
                tfp.layers.IndependentNormal.params_size(self.condition_size),
                activation=tf.nn.softmax,
            )(cond_h_1)
            conditions_op = tfp.layers.IndependentNormal(self.condition_size)(
                conditions_op
            )
        return conditions_op

    @lazy_scope
    def loss_shift(self):
        loss_shift = tf.reduce_mean(
            tf.squared_difference(self.shift_x_trans_mat, self.y_trans_mat)
        )
        return loss_shift

    @lazy_scope
    def loss_conditional(self):
        logits_y_hat = self.condition_y(self.shift_x_trans_mat)
        print("logits_y_hat shape:", logits_y_hat.shape)

        cond_y_hat = tf.one_hot(
            tf.argmax(logits_y_hat, dimension=1), depth=self.condition_size
        )
        print("cond_y_hat shape:", cond_y_hat.shape)

        logits_y = self.condition_y(self.y_trans_mat)
        print("logits_y shape:", logits_y.shape)

        cond_y = tf.one_hot(
            tf.argmax(input=logits_y, axis=1), depth=self.condition_size
        )

        loss_cond = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=cond_y, logits=logits_y_hat
            )
        )
        return loss_cond

    @lazy_scope
    def loss_transition(self):
        """Aggregates the loss_shift and loss_conditional terms into the tranisitonal loss."""
        loss_trans = (
            self.loss_shift + self.conditional_loss_weight * self.loss_conditional
        )
        tf.summary.scalar("loss_shift", self.loss_shift)
        tf.summary.scalar(
            "loss_cond", self.conditional_loss_weight * self.loss_conditional
        )
        tf.summary.scalar("loss_trans", loss_trans)
        return loss_trans

    @lazy_scope
    def condition_pred_y(self):
        logits_y_hat = self.condition_y(self.x_trans_mat)
        cond_y_hat = tf.one_hot(
            tf.argmax(logits_y_hat, dimension=1),
            depth=self.condition_size,
            name="conditions_one_hot",
        )
        return cond_y_hat
