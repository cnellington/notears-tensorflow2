import logging
import numpy as np
import tensorflow as tf
import pdb

from helpers.dir_utils import create_dir
from helpers.tf_utils import print_summary, set_seed
from helpers.analyze_utils import count_accuracy


class NoTears(object):
    """
    NO-TEARS model with augmented Lagrangian gradient-based optimization training
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, use_gpu, seed=8, use_float64=False):
        self.print_summary = print_summary  # Print summary for tensorflow variables

        self.use_gpu = use_gpu
        self.seed = seed
        set_seed(seed)
        self.tf_float_type = tf.dtypes.float64 if use_float64 else tf.dtypes.float32

        # Initializer (for reproducibility)
        tf.keras.backend.clear_session()
        self.initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)

        self.n = None
        self.d = None
        self.l1_lambda = None
        self.rho = None
        self.alpha = None
        self.X = None
        self.W_prime = None

        self._logger.debug('Finished building Tensorflow graph')

        # self.rho = tf.Variable(initial_value=self.initializer(shape=[1], dtype=self.tf_float_type))
        # self.alpha = tf.Variable(initial_value=self.initializer(shape=[1], dtype=self.tf_float_type))
        #
        # self.X = tf.Variable(initial_value=self.initializer(shape=[self.n, self.d], dtype=self.tf_float_type))
        # W = tf.Variable(tf.zeros([self.d, self.d], dtype=self.tf_float_type))
        #
        # self.W_prime = tf.Variable(initial_value=self._preprocess_graph(W), dtype=self.tf_float_type)
        # self.mse_loss = self._get_mse_loss()
        # self.h = self.get_h()
        #
        # self.loss = lambda: 0.5 / self.n * self.mse_loss + self.l1_lambda * tf.norm(tensor=self.W_prime, ord=1) + self.alpha * self.h + 0.5 * self.rho * self.h * self.h

        # self._init_session()
        # self._init_saver()

    @tf.function
    def loss(self):
        mse_loss = self._get_mse_loss()
        h = self.h()
        loss_val = 0.5 / self.n * mse_loss \
            + self.l1_lambda * tf.norm(tensor=self.W_prime, ord=1) \
            + self.alpha * h + 0.5 * self.rho * h * h
        return loss_val

    @tf.function
    def h(self):  # Acyclicity
        return tf.linalg.trace(tf.linalg.expm(self.W_prime * self.W_prime)) - self.d

    def train(self, X, W_true, l1_lambda=0, learning_rate=1e-3,
              graph_thres=0.3, h_thres=0.25, h_tol=1e-8,
              init_rho=1.0, rho_multiply=10.0, rho_thres=1e+12,
              init_iter=3, iter_step=1500, max_iter=20):
        self.X = tf.convert_to_tensor(X, dtype=self.tf_float_type)
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        self.l1_lambda = l1_lambda
        self.rho = init_rho
        self.alpha = 0.0

        W = tf.zeros([self.d, self.d], self.tf_float_type)
        self.W_prime = tf.Variable(self._preprocess_graph(W), dtype=self.tf_float_type, trainable=True)
        train_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # pdb.set_trace()

        self._logger.info('Started training for {} iterations'.format(max_iter))
        for epoch in range(1, max_iter + 1):
            while self.rho < rho_thres:
                self._logger.info('rho {:.3E}, alpha {:.3E}'.format(self.rho, self.alpha))
                h_prev = self.h().numpy()
                self.train_step(train_opt, iter_step)
                print(f"W_prime[0][0]: {self.W_prime[0][0]}")
                if self.h().numpy() > h_thres * h_prev:
                    self.rho *= rho_multiply
                else:
                    break

            self.train_callback(epoch, W_true, graph_thres)
            self.alpha = self.rho * self.h()

            if self.h() < h_tol and epoch > init_iter:
                self._logger.info('Early stopping at {}-th iteration'.format(epoch))
                break

        return self.W_prime.numpy()

    def train_step(self, train_opt, iter_step):
        for _ in range(iter_step):
            with tf.GradientTape() as tape:
                loss = self.loss()
            vars = [self.W_prime]
            grads = tape.gradient(loss, vars)
            train_opt.apply_gradients(zip(grads, vars))

    def train_callback(self, epoch, W_true, graph_thres):
        # Evaluate the learned W in each iteration after thresholding
        W_thresholded = np.copy(self.W_prime.numpy())
        W_thresholded[np.abs(W_thresholded) < graph_thres] = 0
        results_thresholded = count_accuracy(W_true, W_thresholded)

        self._logger.info(
            '[Iter {}] loss {:.3E}, mse {:.3E}, acyclic {:.3E}, shd {}, tpr {:.3f}, fdr {:.3f}, pred_size {}'.format(
                epoch, self.loss(), self._get_mse_loss(), self.h(), results_thresholded['shd'], results_thresholded['tpr'],
                results_thresholded['fdr'], results_thresholded['pred_size']
            )
        )

        # Save the raw estimated graph in each iteration
        # create_dir('{}/raw_estimated_graph'.format(output_dir))
        # np.save('{}/raw_estimated_graph/graph_iteration_{}.npy'.format(output_dir, epoch), W_est)

    def _preprocess_graph(self, W):
        # Mask the diagonal entries of graph
        return tf.linalg.set_diag(W, tf.zeros(W.shape[0], dtype=self.tf_float_type))

    @tf.function
    def _get_mse_loss(self):
        X_prime = tf.matmul(self.X, self.W_prime)
        return tf.square(tf.linalg.norm(tensor=self.X - X_prime))

    # def _init_session(self):
    #     if self.use_gpu:
    #         # Use GPU
    #         self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    #             gpu_options=tf.compat.v1.GPUOptions(
    #                 per_process_gpu_memory_fraction=0.5,
    #                 allow_growth=True,
    #             )
    #         ))
    #     else:
    #         self.sess = tf.compat.v1.Session()

    # def _init_saver(self):
    #     self.saver = tf.compat.v1.train.Saver()

    # def save(self, model_dir):
    #     create_dir(model_dir)
    #     self.saver.save(self.sess, '{}/model'.format(model_dir))

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
    print('model.train_op: {}'.format(model.train_op))
