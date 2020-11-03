import tensorflow as tf
import tensorflow.contrib.layers as tcl

EPSILON = 1e-6


def log_sum_exp(logits):
    feat_dims = logits.get_shape().as_list()[1]
    max_logits = tf.reduce_max(logits, 1)
    return tf.log(tf.reduce_sum(tf.exp(logits - tf.tile(tf.expand_dims(max_logits, 1), [1, feat_dims])), 1)) + max_logits


class SVAEMNIST(object):
    def __init__(self, dim_z=32, classes=10):
        self.dim_z = dim_z
        self.classes = classes

        self._encoder_name = 'Encoder'
        self._generator_name = 'Generator'
        self._classifier_name = 'Classifier'
        self._discriminator_name = 'Discriminator'

        self._generate = tf.make_template(
            self._generator_name,
            self._generator)
        self._encode = tf.make_template(
            self._encoder_name,
            self._encoder)
        self._classify = tf.make_template(
            self._classifier_name,
            self._classifier)
        self._discriminate = tf.make_template(
            self._discriminator_name,
            self._discriminator)

    def _encoder(self, x):
        dim_z = self.dim_z
        elu = tf.nn.elu
        x = tcl.conv2d(x, 16, kernel_size=3, activation_fn=elu)
        x = tcl.conv2d(x, 32, kernel_size=3, stride=2, activation_fn=elu)
        x = tcl.conv2d(x, 32, kernel_size=3, activation_fn=elu)
        x = tcl.conv2d(x, 64, kernel_size=3, stride=2, activation_fn=elu)
        x = tcl.flatten(x)
        x = tcl.fully_connected(x, 128, activation_fn=tf.nn.tanh)
        x = tcl.fully_connected(x, dim_z * 2, activation_fn=None)

        mu = x[:, :dim_z]
        sigma = EPSILON + tf.nn.softplus(x[:, dim_z:])
        return mu, sigma

    def _generator(self, z):
        elu = tf.nn.elu
        x = tcl.fully_connected(z, 128, activation_fn=elu)
        x = tcl.fully_connected(x, 7 * 7 * 64, activation_fn=elu)
        x = tf.reshape(x, (-1, 7, 7, 64))
        x = tf.image.resize_bilinear(x, (14, 14))
        x = tcl.conv2d(x, 32, kernel_size=3, stride=1, activation_fn=elu)
        x = tcl.conv2d(x, 32, kernel_size=3, stride=1, activation_fn=elu)
        x = tf.image.resize_bilinear(x, (28, 28))
        x = tcl.conv2d(x, 16, kernel_size=3, stride=1, activation_fn=elu)
        x = tcl.conv2d(x, 1, kernel_size=3, stride=1, activation_fn=tf.nn.sigmoid)
        return x

    def _classifier(self, z):
        classes = self.classes

        x = tcl.fully_connected(z, 256, activation_fn=tf.nn.elu)
        x = tcl.fully_connected(x, 128, activation_fn=tf.nn.elu)
        x = tcl.fully_connected(x, classes, activation_fn=None)

        return x

    def _discriminator(self, z):
        x = tcl.fully_connected(z, 256, activation_fn=tf.nn.elu)
        x = tcl.fully_connected(x, 128, activation_fn=tf.nn.elu)
        x = tcl.fully_connected(x, 64, activation_fn=tf.nn.elu)
        x = tcl.fully_connected(x, 1, activation_fn=None)

        return x

    def KL_divergence(self, mu, sigma):
        return tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(EPSILON + tf.square(sigma)) - 1, 1))

    def build(self, x, y, batch_size, learning_rate):

        # ==== conditional variational auto-encoder
        mu, sigma = self.encode(x)
        z = mu + sigma * tf.random_normal(tf.shape(mu))
        z_sample = tf.random_normal(tf.shape(mu))
        dis_z = self.discriminate(z)
        dis_z_sample = self.discriminate(z_sample)

        x_hat = self.decode(z)
        y_hat = self.classify(z)

        z_fake = tf.random_normal(tf.shape(mu))
        x_fake_hat = self.decode(z_fake)

        # ==== variables
        variables = dict()

        var_enc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._encoder_name)
        var_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._generator_name)
        var_cla = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._classifier_name)
        var_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._discriminator_name)

        variables['enc'] = var_enc
        variables['gen'] = var_gen
        variables['cla'] = var_cla
        variables['dis'] = var_dis

        # ==== define losses
        loss = dict()
        x_hat_clipped = tf.clip_by_value(x_hat, EPSILON, 1-EPSILON)

        # Loss stage1: SAE
        loss['d_cla'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
        loss['d_KL'] = self.KL_divergence(mu, sigma)
        loss['g_rec'] = -tf.reduce_sum(x * tf.log(x_hat_clipped) + (1-x)*tf.log(1-x_hat_clipped)) / float(batch_size)

        loss['SAE'] = loss['d_cla'] + loss['d_KL'] + loss['g_rec']

        # Loss stage2: Gan
        loss['dis'] = tf.reduce_mean(0.5 * tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_z, labels=tf.ones_like(dis_z)) +
                                     0.5 * tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_z_sample, labels=tf.zeros_like(dis_z_sample)))

        # ==== define optimizer
        optimizer = tf.train.AdamOptimizer
        optim = dict()

        # ---- svae optimizer
        optim['SAE'] = optimizer(learning_rate=learning_rate, beta1=0.5).minimize(loss['SAE'], var_list=var_enc + var_gen + var_cla)

        # ---- gan optimizer
        optim['DIS'] = optimizer(learning_rate=learning_rate, beta1=0.5).minimize(loss['dis'], var_list=var_dis)

        # ==== define metrics
        metric = dict()

        # ---- svae metric
        metric['acc'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1)), tf.float32))
        metric['x_hat'] = x_hat
        metric['latent'] = mu
        metric['x_fake_hat'] = x_fake_hat
        metric['prob'] = tf.nn.softmax(logits=y_hat)

        # ---- gan metric
        metric['acc_dis_true'] = tf.reduce_mean(tf.cast(dis_z >= 0.5, tf.float32))
        metric['acc_dis_fake'] = tf.reduce_mean(tf.cast(dis_z_sample < 0.5, tf.float32))
        return loss, optim, metric, variables

    def encode(self, x):
        return self._encode(x)

    def decode(self, z):
        return self._generate(z)

    def classify(self, z):
        return self._classify(z)

    def autoencoder_x(self, x):
        mu, sigma = self.encode(x)
        x_hat = self.decode(mu)
        return x_hat

    def discriminate(self, z):
        return self._discriminate(z)


class MNIST_MLP:
    def __init__(self, classes=10):
        self.classes = classes

        self._classifier_name = 'MNIST_MLP'

        self._classify = tf.make_template(
            self._classifier_name,
            self._classifier)

    def _classifier(self, x):
        classes = self.classes

        # x = tcl.flatten(x)
        # x = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu)
        # x = tcl.fully_connected(x, classes, activation_fn=None)
        
        relu = tf.nn.relu
        x = tcl.conv2d(x, 32, kernel_size=2, activation_fn=relu)
        x = tcl.max_pool2d(x, kernel_size=2)
        x = tcl.conv2d(x, 64, kernel_size=2, activation_fn=relu)
        last_conv = tcl.max_pool2d(x, kernel_size=2)
        x = tcl.avg_pool2d(last_conv, kernel_size=7)
        x = tcl.flatten(x)
        x = tcl.fully_connected(x, classes, activation_fn=None)

        return x, last_conv

    def build(self, x, y, batch_size, learning_rate):
        y_hat, last_conv = self.classify(x)

        variables = dict()
        var_cla = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._classifier_name)
        variables['cla'] = var_cla

        loss = dict()
        loss['cla'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

        optim = dict()
        optim['cla'] = tf.train.AdamOptimizer(learning_rate).minimize(loss['cla'])

        metric = dict()
        metric['acc'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1)), tf.float32))
        metric['prob'] = tf.nn.softmax(logits=y_hat)
        return loss, optim, metric, variables, last_conv

    def classify(self, x):
        return self._classify(x)