import tensorflow as tf
import numpy as np
import mnist_data
import plot_utils
import os
import mnist_model
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scipy.io as scio

IMAGE_SIZE = 28
np.random.seed(996)
tf.set_random_seed(996)


def train(args):
    classes = args.classes
    dim_z = args.dim_z
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    RESULT_DIR = args.results_path
    DATASET = 'mnist-dim_z{}'.format(dim_z)
    TEMP_DIR = './tmp'

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(os.path.join(RESULT_DIR, DATASET)):
        os.makedirs(os.path.join(RESULT_DIR, DATASET))

    x_train, y_train, x_test, y_test = mnist_data.load_mnist(reshape=True, twoclass=None, binary=True, onehot=True)

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input_img')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='input_label')

    svae = mnist_model.SVAEMNIST(dim_z, classes)
    loss, optim, metric, variables = svae.build(x, y, batch_size, learning_rate)

    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    total_train_batch = int(n_train_samples / batch_size)
    total_test_batch = int(n_test_samples / batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if True:  # stage 1: train the supervised variational autoencoder
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

            # restore training
            # saver.restore(sess, os.path.join(RESULT_DIR, DATASET, "{}_epoch_80.data".format(DATASET)))

            for epoch in range(0, epochs+1):
                # Random shuffling
                indexes = np.arange(0, n_train_samples)
                np.random.shuffle(indexes)
                x_train = x_train[indexes, ...]
                y_train = y_train[indexes, ...]
                for i in range(total_train_batch):
                    # Compute the offset of the current minibatch in the data.
                    offset = (i * batch_size) % n_train_samples
                    batch_xs = x_train[offset:(offset + batch_size), ...]
                    batch_ys = y_train[offset:(offset + batch_size), ...]

                    feed_dict = {x: batch_xs, y: batch_ys}
                    # update sae
                    _, = sess.run([optim['SAE']], feed_dict=feed_dict)

                    if i % 100 == 0:
                        loss_tot, d_cla, d_KL, g_rec, acc = sess.run(
                            [loss['SAE'], loss['d_cla'],  loss['d_KL'], loss['g_rec'], metric['acc']],
                            feed_dict=feed_dict)
                        msg = '[stage1] epoch:{}/{} step:{}/{} '.format(epoch, epochs, i, total_train_batch)
                        msg += 'loss_tot:{:.4}, d_cla:{:.4}, d_KL:{:.4}, g_rec:{:.4}, acc{:.4}'.format(loss_tot, d_cla, d_KL, g_rec, acc)
                        print(msg)

                # validate after each epoch
                batch_xs = x_test[0:100, ...]
                x_rec, x_gen = sess.run([metric['x_hat'], metric['x_fake_hat']], feed_dict={x: batch_xs})
                PAE = plot_utils.Plot_Adversarial_Example(TEMP_DIR)
                PAE.add_images(list(mnist_data.deprocess(batch_xs)))
                PAE.save_images(name="{}_ori.png".format(str(epoch).zfill(3)))
                PAE.clear()
                PAE.add_images(list(mnist_data.deprocess(x_rec)))
                PAE.save_images(name="{}_rec.png".format(str(epoch).zfill(3)))
                PAE.clear()
                PAE.add_images(list(mnist_data.deprocess(x_gen)))
                PAE.save_images(name="{}_gen.png".format(str(epoch).zfill(3)))

                val_loss_tot, val_d_cla, val_d_KL, val_g_rec, val_acc = 0.0, 0.0, 0.0, 0.0, 0.0

                for i in range(total_test_batch):
                    offset = (i * batch_size) % n_test_samples
                    batch_xs = x_test[offset:(offset + batch_size), ...]
                    batch_ys = y_test[offset:(offset + batch_size), ...]
                    feed_dict = {x: batch_xs, y: batch_ys}
                    loss_tot, d_cla, d_KL, g_rec, acc = sess.run(
                        [loss['SAE'], loss['d_cla'], loss['d_KL'], loss['g_rec'], metric['acc']],
                        feed_dict=feed_dict)

                    val_loss_tot += loss_tot
                    val_d_cla += d_cla
                    val_d_KL += d_KL
                    val_g_rec += g_rec
                    val_acc += acc

                msg = 'epoch:{}/{} '.format(epoch, epochs)
                msg += 'val_loss_tot:{:.4}, val_d_cla:{:.4}, val_d_KL:{:.4}, val_g_rec:{:.4}'.format(val_loss_tot / total_test_batch,
                                                                                                     val_d_cla / total_test_batch,
                                                                                                     val_d_KL / total_test_batch,
                                                                                                     val_g_rec / total_test_batch)
                msg += 'val_acc:{:.4}'.format(val_acc / total_test_batch)
                print(msg)
                if epoch % 10 == 0:
                    saver.save(sess, os.path.join(RESULT_DIR, DATASET, "{}_epoch_{}.data".format(DATASET, epoch)))
            saver.save(sess, os.path.join(RESULT_DIR, DATASET, "{}_stage1.data".format(DATASET)))

        if True:  # stage2: train a discriminator to estimate the manifold in the gaussian latent space
            saver_loader = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['enc']+variables['gen']+variables['cla'])
            saver_loader.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage1.data'.format(DATASET)))
            saver_writer = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['dis'])

            for epoch in range(0, epochs+1):
                # Random shuffling
                indexes = np.arange(0, n_train_samples)
                np.random.shuffle(indexes)
                x_train = x_train[indexes, ...]
                y_train = y_train[indexes, ...]
                for i in range(total_train_batch):
                    # Compute the offset of the current minibatch in the data.
                    offset = (i * batch_size) % n_train_samples
                    batch_xs = x_train[offset:(offset + batch_size), ...]
                    batch_ys = y_train[offset:(offset + batch_size), ...]

                    feed_dict = {x: batch_xs, y: batch_ys}
                    # update sae
                    _ = sess.run(optim['DIS'], feed_dict=feed_dict)

                    if i % 100 == 0:
                        loss_dis, acc_dis_true, acc_dis_fake = sess.run(
                            [loss['dis'], metric['acc_dis_true'],  metric['acc_dis_fake']],
                            feed_dict=feed_dict)
                        msg = '[stage2] epoch:{}/{} step:{}/{} '.format(epoch, epochs, i, total_train_batch)
                        msg += 'loss_dis:{:.4}, acc_dis_true:{:.4}, acc_dis_fake:{:.4}'.format(loss_dis, acc_dis_true, acc_dis_fake)
                        print(msg)

                if epoch % 10 == 0:
                    saver_writer.save(sess, os.path.join(RESULT_DIR, DATASET, "{}_stage2_epoch_{}.data".format(DATASET, epoch)))
            saver_writer.save(sess, os.path.join(RESULT_DIR, DATASET, "{}_stage2.data".format(DATASET)))


def train_MLP(args):

    classes = args.classes
    dim_z = args.dim_z
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    RESULT_DIR = args.results_path
    DATASET = 'mnist-MLP-dim_z{}'.format(dim_z)
    TEMP_DIR = './tmp'
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(os.path.join(RESULT_DIR, DATASET)):
        os.makedirs(os.path.join(RESULT_DIR, DATASET))
    if not os.path.exists('./CAM'):
        os.makedirs('./CAM')

    x_train, y_train, x_test, y_test = mnist_data.load_mnist(reshape=True, twoclass=None, binary=False, onehot=True)

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input_img')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='input_label')

    machine = mnist_model.MNIST_MLP(classes)
    loss, optim, metric, variables, last_conv = machine.build(x, y, batch_size, learning_rate)

    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    total_train_batch = int(n_train_samples / batch_size)
    total_test_batch = int(n_test_samples / batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ###
        all_vars = tf.global_variables()
        print(all_vars)
        para = all_vars[4]
        gap_1 = all_vars[-4]
        gap_2 = all_vars[-3]
        ###
        
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

        for epoch in range(0, epochs + 1):
            # Random shuffling
            indexes = np.arange(0, n_train_samples)
            np.random.shuffle(indexes)
            x_train = x_train[indexes, ...]
            y_train = y_train[indexes, ...]
            for i in range(total_train_batch):
                offset = (i * batch_size) % n_train_samples
                batch_xs = x_train[offset:(offset + batch_size), ...]
                batch_ys = y_train[offset:(offset + batch_size), ...]

                feed_dict = {x: batch_xs, y: batch_ys}

                # update cla
                sess.run(optim['cla'], feed_dict=feed_dict)

                if i % 100 == 0:
                    d_cla, acc = sess.run(
                        [loss['cla'], metric['acc']],
                        feed_dict=feed_dict)
                    msg = 'epoch:{}/{} step:{}/{} '.format(epoch, epochs, i, total_train_batch)
                    msg += 'cla:{:.4},  acc{:.4}' \
                        .format(d_cla, acc)
                    print(msg)

            # validate after each epoch
            val_d_cla, val_acc = 0.0, 0.0

            for i in range(total_test_batch):
                offset = (i * batch_size) % n_train_samples
                batch_xs = x_test[offset:(offset + batch_size), ...]
                batch_ys = y_test[offset:(offset + batch_size), ...]
                feed_dict = {x: batch_xs, y: batch_ys}
                d_cla, acc = sess.run(
                    [loss['cla'], metric['acc']],
                    feed_dict=feed_dict)
                
                val_d_cla += d_cla
                val_acc += acc

            msg = 'epoch:{}/{} '.format(epoch, epochs)
            msg += 'val_cla:{:.4}'.format(val_d_cla / total_test_batch)
            msg += 'val_acc:{:.4}'.format(val_acc / total_test_batch)
            print(msg)
            last_layer, paras, gap_para_1, gap_para_2 = sess.run([last_conv, para, gap_1, gap_2], feed_dict=feed_dict)
            if epoch == epochs:
                print('----------------------------')
                print(last_layer.shape)
                print(paras.shape)
                print(gap_para_1.shape)
                print(gap_para_2.shape)
                np.save('./CAM/x', batch_xs)
                np.save('./CAM/y', batch_ys)
                np.save('./CAM/last_layer', last_layer)
                np.save('./CAM/paras', paras)
                np.save('./CAM/fc_para_1', gap_para_1)
                np.save('./CAM/fc_para_2', gap_para_2)
                print('----------------------------')
            if epoch % 10 == 0:
                saver.save(sess, os.path.join(RESULT_DIR, DATASET, "{}_epoch_{}.data".format(DATASET, epoch)))
        saver.save(sess, os.path.join(RESULT_DIR, DATASET, "{}.data".format(DATASET)))
        

def transition(args):
    test_img_index = args.test_index  # modify this to test on different images  
    target_img_label = args.target_label    # modify this to transition the current image to different label

    classes = args.classes
    dim_z = args.dim_z
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    RESULT_DIR = args.results_path
    DATASET = 'mnist-dim_z{}'.format(dim_z)
    TEMP_DIR = './tmp'
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(os.path.join(RESULT_DIR, DATASET)):
        os.makedirs(os.path.join(RESULT_DIR, DATASET))

    # load dataset
    x_train, y_train, x_test, y_test = mnist_data.load_mnist(reshape=True, twoclass=None, binary=True, onehot=True)

    test_img = x_test[test_img_index, ...]
    test_label = np.argmax(y_test[test_img_index, ...]).squeeze()

    # target_img_label = (test_label + 1) if test_label != 9 else 0
    print('target label is ', target_img_label)
    # choose test image and its target label
    target_label = np.zeros(shape=[1, classes])
    target_label[0, target_img_label] = 1

    # build the testing network
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input_img')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='input_label')

    z = tf.get_variable('z_var', shape=[1, dim_z], dtype=tf.float32)

    svae = mnist_model.SVAEMNIST(dim_z, classes)
    loss, optim, metric, variables = svae.build(x, y, batch_size, learning_rate)

    logits = svae.classify(z)
    x_hat = svae.decode(z)

    dis_logits = svae.discriminate(z)
    l_dis = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits, labels=tf.ones_like(dis_logits)))

    l_cla = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    l_tot = l_cla + 0.01 * l_dis + tf.norm(z) * 0.0005

    # using adam optimizer to manipulate the latent space
    opt = tf.train.AdamOptimizer(5e-3).minimize(l_tot, var_list=[z])

    assign_op = tf.assign(z, metric['latent'])

    with tf.Session() as sess:

        svae_saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['enc'] + variables['gen'] + variables['cla'])
        mc_gan_saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['dis'])
        sess.run(tf.global_variables_initializer())

        # load the parameters of svae
        svae_saver.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage1.data'.format(DATASET)))

        # load the parameters of discrminator on manifold
        mc_gan_saver.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage2.data'.format(DATASET)))

        # initialize z_var
        _ = sess.run(assign_op, feed_dict={x: np.expand_dims(test_img, 0)})

        # plot utils
        PAE = plot_utils.Plot_Adversarial_Example(os.path.join(RESULT_DIR, 'images/'), n_img_x=30)
        PAE.add_image(mnist_data.deprocess(test_img).squeeze())
        loss_cla_buf = []
        dis_buf = []
        verbose_period = 10
        iterations = 20000
        for i in range(iterations+1):
            if i % verbose_period == 0: 
                x_rec, loss_cla, dis = sess.run([x_hat, l_cla, l_dis], feed_dict={y: target_label})
                PAE.add_image(mnist_data.deprocess(x_rec).squeeze())
                loss_cla_buf.append(np.log(loss_cla + 1e-5))
                dis_buf.append(np.log(dis + 1e-5))

            _, loss_tot, loss_cla, dis, val_z = sess.run([opt, l_tot, l_cla, l_dis, z], feed_dict={y: target_label})
            if i % verbose_period == 0:
                msg = 'step {}/{}, loss: {:.6}, loss_cla:{:.6}, dis: {:.6}, max_z: {:.4}'\
                    .format(i, iterations, loss_tot, loss_cla, dis, np.max(np.abs(val_z)))
                print(msg)


        IA = plot_utils.Image_Animation(os.path.join(RESULT_DIR, 'images/'), PAE.img_list)
        IA.save_animation('[{}]{}_to_{}.mp4'.format(test_img_index, test_label, target_img_label))

        PAE.save_images('[{}]{}_to_{}.rec.jpg'.format(test_img_index, test_label, target_img_label))
        plt.plot(np.arange(0, verbose_period*len(loss_cla_buf), verbose_period), loss_cla_buf, c='r', label='oracle classifier loss')
        plt.plot(np.arange(0, verbose_period*len(dis_buf), verbose_period), dis_buf, c='b', label='discriminator loss')
        plt.xlabel('steps')
        plt.ylabel('log loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(RESULT_DIR, 'images/', '[{}]loss_{}_to_{}.jpg'.format(test_img_index, test_label, target_img_label)))

        print('transition outputs to {}'.format(os.path.join(RESULT_DIR, 'images/')))



def attack(args):  # attack mlp on mnist
    test_img_index = args.test_index  # modify this to test on different images
    target_img_label = args.target_label # modify this to attack the current image to different label

    classes = args.classes
    dim_z = args.dim_z
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    RESULT_DIR = args.results_path
    DATASET = 'mnist-dim_z{}'.format(dim_z)
    MLP = 'mnist-MLP-dim_z{}'.format(dim_z)
    TEMP_DIR = './tmp'

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(os.path.join(RESULT_DIR, DATASET)):
        os.makedirs(os.path.join(RESULT_DIR, DATASET))

    # load mnist data
    x_train, y_train, x_test, y_test = mnist_data.load_mnist(reshape=True, twoclass=None, binary=True, onehot=True)

    test_label = np.argmax(y_test[test_img_index, ...]).squeeze()
    # target_img_label = (test_label + 1) if test_label < 9 else 0

    test_img = x_test[test_img_index, ...]
    true_label = np.zeros(shape=[1, classes])
    true_label[0, test_label] = 1

    target_label = np.zeros(shape=[1, classes])
    target_label[0, target_img_label] = 1

    # build the testing network
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input_img')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='input_label')
    y_true = tf.placeholder(tf.float32, shape=[None, classes], name='true_label')
    k_t = tf.placeholder(tf.float32, shape=(), name='k_t')
    val_kt = 0

    z = tf.get_variable('z_var', shape=[1, dim_z], dtype=tf.float32)

    svae = mnist_model.SVAEMNIST(dim_z, classes)
    loss, optim, metric, variables = svae.build(x, y, batch_size, learning_rate)

    logits = svae.classify(z)
    x_hat = svae.decode(z)

    # build the MLP on mnist
    MLP_model = mnist_model.MNIST_MLP(classes)
    MLP_loss, MLP_optim, MLP_metric, MLP_variables, last_layer = MLP_model.build(x_hat, y_true, batch_size, learning_rate)

    dis_logits = svae.discriminate(z)
    l_dis = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits, labels=tf.ones_like(dis_logits)))

    l_cla = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    l_tot = l_cla + 0.01 * l_dis + tf.norm(z) * 0.0001 + MLP_loss['cla'] * k_t

    opt = tf.train.AdamOptimizer(5e-3).minimize(l_tot, var_list=[z])

    assign_op = tf.assign(z, metric['latent'])
    
    with tf.Session() as sess:

        svae_saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['enc'] + variables['gen'] + variables['cla'])
        mc_gan_saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['dis'])
        MLP_saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=MLP_variables['cla'])
        sess.run(tf.global_variables_initializer())
        svae_saver.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage1.data'.format(DATASET)))
        mc_gan_saver.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage2.data'.format(DATASET)))
        MLP_saver.restore(sess, os.path.join(RESULT_DIR, MLP, '{}.data'.format(MLP)))
        # initialize z_var
        _ = sess.run(assign_op, feed_dict={x: np.expand_dims(test_img, 0)})
        ### modify
        all_vars = tf.global_variables()
        print(all_vars)
        para = all_vars[123]
        # for i in range(len(all_vars)):
        #    if(all_vars[i].name == 'MNIST_MLP/fully_connected/weights:0'):
        #        print('--------')
        #        print(i)
        ###
        # plot utils
        PAE = plot_utils.Plot_Adversarial_Example(os.path.join(RESULT_DIR, 'images/'), n_img_x=30)
        PAE.add_image(mnist_data.deprocess(test_img).squeeze())
        verbose_period = 10
        loss_cla_buf = []
        dis_buf = []
        weak_loss_buf = []
        iteration = 20000
        # iteration = 1000
        ### modify
        img = []
        salient_map = []
        prob = []
        ###
        for i in range(iteration+1):
            feed_dict = {y: target_label, y_true: true_label, k_t: val_kt}
            ### modify
            if i % 100 == 0:
                x_rec, last_conv, paras, pred_prob = sess.run([x_hat, last_layer, para, MLP_metric['prob']], feed_dict=feed_dict)
                img.append(x_rec)
                salient_map.append(last_conv)
                prob.append(pred_prob)
            ###
            if i % verbose_period == 0:
                x_rec, loss_cla, dis, weak_prob, loss_weak = sess.run([x_hat, l_cla, l_dis, MLP_metric['prob'], MLP_loss['cla']], feed_dict=feed_dict)

                PAE.add_image(mnist_data.deprocess(x_rec).squeeze())
                loss_cla_buf.append(np.log(loss_cla + 1e-5))
                dis_buf.append(np.log(dis + 1e-5))
                weak_loss_buf.append(np.log(loss_weak+1e-5))

            _, loss_tot, loss_cla, val_dis, val_z, val_loss_weak, weak_prob = sess.run([
                opt,
                l_tot,
                l_cla,
                l_dis,
                z,
                MLP_loss['cla'],
                MLP_metric['prob']], feed_dict=feed_dict)
            val_kt += 1e-4*(val_loss_weak*1e-3-loss_cla+np.maximum(val_loss_weak-0.01, 0))
            val_kt = np.clip(val_kt, 0.0, 0.001)
            if i % verbose_period == 0:
                msg = 'step {}/{}, loss: {:.4}, loss_cla:{:.6}, val_dis: {:.6}, max_z: {:.4}, loss_MLP: {:.3}, MLP_pred/prob: {}/{:.4}, kt:{:.4}' \
                    .format(i, iteration, loss_tot, loss_cla, val_dis, np.max(np.abs(val_z)), val_loss_weak, np.argmax(weak_prob), np.max(weak_prob), val_kt)
                print(msg)
        ###
        print(np.array(img).shape)
        print(paras.shape)
        print(np.array(salient_map).shape)
        print(np.array(prob).shape)
        np.save('./CAM/attack_img_{}to{}'.format(test_label, target_img_label), np.array(img))
        np.save('./CAM/paras', paras)
        np.save('./CAM/last_layer_{}to{}'.format(test_label, target_img_label), np.array(salient_map))
        np.save('./CAM/pred_prob_{}to{}'.format(test_label, target_img_label), np.array(prob))
        ###
        IA = plot_utils.Image_Animation(os.path.join(RESULT_DIR, 'images/'), PAE.img_list)
        IA.save_animation('[{}]attack_{}_to_{}.mp4'.format(test_img_index, test_label, target_img_label))
        PAE.save_images('[{}]attack_rec_{}_to_{}.jpg'.format(test_img_index, test_label, target_img_label))

        plt.figure(figsize=(10,5))
        plt.ylim(-15, 5)
        plt.plot(np.arange(0, len(loss_cla_buf)*verbose_period, verbose_period), loss_cla_buf, c='r', label="oracle $f_2$ loss@" + "$y'$=5")#.format(target_img_label)
        plt.plot(np.arange(0, len(dis_buf)*verbose_period, verbose_period), dis_buf, c='b', label='discriminator $f_{dis}$ loss')
        plt.plot(np.arange(0, len(weak_loss_buf)*verbose_period, verbose_period), weak_loss_buf, c='g', label="MLP $f_1$ loss @" + "$y$={}".format(test_label))
        # plt.plot(np.arange(0, len(loss_tot_buf)*verbose_period, verbose_period), loss_tot_buf, c='black', label='total loss $J_{SA}$')
        plt.xlabel('iteration steps')
        plt.ylabel('log loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(RESULT_DIR, 'images/', '[{}]attack_loss_{}_to_{}.jpg'.format(test_img_index, test_label, target_img_label)))

        # output the final image
        prob, x_img = sess.run([MLP_metric['prob'], x_hat])
        x_img_out = mnist_data.deprocess(x_img).squeeze()
        final_path = os.path.join(RESULT_DIR, 'images/','[{}]final_{}to{}_{:.3}.jpg').format(test_img_index, prob.argmax(), target_img_label, np.max(prob))
        print('output final adversarial example to:', final_path)
        plot_utils.imsave(final_path, x_img_out)
        
        