import logging
import numpy as np
import tensorflow as tf

from helpers.dir_utils import create_dir
from helpers.tf_utils import print_summary, set_seed
from helpers.analyze_utils import count_accuracy


class NoTears(object):
    """
    NO-TEARS model with augmented Lagrangian gradient-based optimization
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, use_gpu, seed=8, use_float64=False):
        self.print_summary = print_summary  # Variable summary not supported with eager execution

        self.use_gpu = use_gpu
        self.seed = seed
        set_seed(seed)
        self.tf_float_type = tf.dtypes.float64 if use_float64 else tf.dtypes.float32

        # Placeholder vars
        tf.keras.backend.clear_session()
        self.n = None
        self.d = None
        self.l1_lambda = None
        self.rho = None
        self.alpha = None
        self.X = None
        self.W_prime = None

        self._logger.debug('Finished building NoTears model')
        self.scope = self._init_gpu()  # GPU execution not implemented

    @tf.function
    def loss(self):
        mse_loss = self.mse_loss()
        h = self.h()
        return 0.5 / self.n * mse_loss \
            + self.l1_lambda * tf.norm(tensor=self.W_prime, ord=1) \
            + self.alpha * h + 0.5 * self.rho * h * h

    @tf.function
    def h(self):  # Acyclicity
        return tf.linalg.trace(tf.linalg.expm(self.W_prime * self.W_prime)) - self.d

    @tf.function
    def mse_loss(self):
        X_prime = tf.matmul(self.X, self.W_prime)
        return tf.square(tf.linalg.norm(tensor=self.X - X_prime))

    def train(self, X, W_true, output_dir, l1_lambda=0, learning_rate=1e-3,
              graph_thres=0.3, h_thres=0.25, h_tol=1e-8,
              init_rho=1.0, rho_multiply=10.0, rho_thres=1e+12,
              init_iter=3, iter_step=1500, max_iter=20):
        self.X = tf.convert_to_tensor(X, dtype=self.tf_float_type)
        self.n = tf.constant(self.X.shape[0], dtype=self.tf_float_type)
        self.d = tf.constant(self.X.shape[1], dtype=self.tf_float_type)
        self.l1_lambda = tf.constant(l1_lambda, dtype=self.tf_float_type)
        self.rho = tf.Variable(init_rho, dtype=self.tf_float_type)
        self.alpha = tf.Variable(0.0, dtype=self.tf_float_type)

        W = tf.zeros([self.d, self.d], self.tf_float_type)
        self.W_prime = tf.Variable(self._preprocess_graph(W), dtype=self.tf_float_type, trainable=True)
        train_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self._logger.info('Started training for {} iterations'.format(max_iter))
        h, h_new = np.inf, np.inf
        for epoch in range(1, max_iter + 1):
            while self.rho.numpy() < rho_thres:
                self._logger.info('rho {:.3E}, alpha {:.3E}'.format(self.rho.numpy(), self.alpha.numpy()))
                for _ in range(iter_step):  # Train step
                    train_opt.minimize(self.loss, [self.W_prime])
                h_new = self.h().numpy()
                if h_new > h_thres * h:
                    self.rho.assign(self.rho.numpy() * rho_multiply)
                else:
                    break

            self.train_callback(epoch, W_true, graph_thres, output_dir)
            h = h_new
            self.alpha.assign_add(self.rho.numpy() * h)

            if h < h_tol and epoch > init_iter:
                self._logger.info('Early stopping at {}-th iteration'.format(epoch))
                break

        # No sessions in TF2 Eager Execution, W_prime is the only output necessary
        return self.W_prime.numpy()

    def train_callback(self, epoch, W_true, graph_thres, output_dir):
        # Evaluate the learned W in each iteration after thresholding
        W_thresholded = np.copy(self.W_prime.numpy())
        W_thresholded[np.abs(W_thresholded) < graph_thres] = 0
        results_thresholded = count_accuracy(W_true, W_thresholded)

        self._logger.info(
            '[Iter {}] loss {:.3E}, mse {:.3E}, acyclic {:.3E}, shd {}, tpr {:.3f}, fdr {:.3f}, pred_size {}'.format(
                epoch, self.loss(), self.mse_loss(), self.h(), results_thresholded['shd'], results_thresholded['tpr'],
                results_thresholded['fdr'], results_thresholded['pred_size']
            )
        )

        # Save the raw estimated graph in each iteration
        create_dir('{}/raw_estimated_graph'.format(output_dir))
        np.save('{}/raw_estimated_graph/graph_iteration_{}.npy'.format(output_dir, epoch), self.W_prime.numpy())

    def _preprocess_graph(self, W):
        # Mask the diagonal entries of graph
        return tf.linalg.set_diag(W, tf.zeros(W.shape[0], dtype=self.tf_float_type))

    def _init_gpu(self):
        if self.use_gpu:
            # Use GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return tf.distribute.MirroredStrategy(devices=gpus)
            except RuntimeError as e:
                print(e)

    @property
    def logger(self):
        try:
            return self._logger
        except:
            raise NotImplementedError('self._logger does not exist!')


if __name__ == '__main__':
    n, d = 3000, 20
    model = NoTears(False, n, d)
    # model.print_summary(print)

    print()
    print('model.W_prime: {}'.format(model.W_prime))
    print('model.alpha: {}'.format(model.alpha))
    print('model.mse_loss: {}'.format(model.mse_loss))
    print('model.h: {}'.format(model.h))
    print('model.loss: {}'.format(model.loss))
