# ---------------------------------------------------
#   code credits: https://github.com/CQFIO/PhotographicImageSynthesis
# ---------------------------------------------------


from __future__ import division

from FetchManager import *
from config import *
from CX_helper import *
from model import *


sess = tf.Session()
sp = config.TRAIN.sp

# ---------------------------------------------------
#                      graph
# ---------------------------------------------------
with tf.variable_scope(tf.get_variable_scope()):
    input_A = tf.placeholder(tf.float32, [None, None, None, 3])
    input_B = tf.placeholder(tf.float32, [None, None, None, 3])
    input_A_test = tf.placeholder(tf.float32, [None, None, None, 3])
    input_image_A, real_image_B = helper.random_crop_together(input_A, input_B, [2, config.TRAIN.resize[0], config.TRAIN.resize[1], 3])#
    with tf.variable_scope("g") as scope:
        generator = recursive_generator(input_image_A, sp)
        scope.reuse_variables()
        generator_test = recursive_generator(input_A_test, sp)
    weight = tf.placeholder(tf.float32)
    vgg_real = build_vgg19(real_image_B)
    vgg_fake = build_vgg19(generator, reuse=True)
    vgg_input = build_vgg19(input_image_A, reuse=True)


    ## --- contextual style---
    if config.W.CX > 0:
        CX_loss_list = [w * CX_loss_helper(vgg_real[layer], vgg_fake[layer], config.CX)
                        for layer, w in config.CX.feat_layers.items()]
        CX_style_loss = tf.reduce_sum(CX_loss_list)
        CX_style_loss *= config.W.CX
    else:
        CX_style_loss = zero_tensor

    ## --- contextual content---
    if config.W.CX_content > 0:
        CX_loss_content_list = [w * CX_loss_helper(vgg_input[layer], vgg_fake[layer], config.CX)
                                for layer, w in config.CX.feat_content_layers.items()]
        CX_content_loss = tf.reduce_sum(CX_loss_content_list)
        CX_content_loss *= config.W.CX_content
    else:
        CX_content_loss = zero_tensor

    ## --- total loss ---
    G_loss = CX_style_loss + CX_content_loss


# create the optimization
lr = tf.placeholder(tf.float32)
var_list = [var for var in tf.trainable_variables() if var.name.startswith('g/g_')]
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=var_list)
saver = tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())


# load from checkpoint if exist
def load(dir):
    ckpt = tf.train.get_checkpoint_state(dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    return ckpt


ckpt = load(config.TRAIN.out_dir)


# ---------------------------------------------------
#                      train
# ---------------------------------------------------
if config.TRAIN.is_train:
    file_list = os.listdir(config.base_dir + config.TRAIN.A_data_dir)
    val_file_list = os.listdir(config.base_dir + config.VAL.A_data_dir)
    file_list = np.random.permutation(file_list)
    assert len(file_list) > 0
    train_file_list = file_list[0::config.TRAIN.every_nth_frame]
    val_file_list = val_file_list[0::config.VAL.every_nth_frame]
    g_loss = np.zeros(len(train_file_list), dtype=float)
    fetcher = FetchManager(sess, [G_opt, G_loss])
    B_file_name = config.single_image_B_file_name
    B_image = helper.read_image(B_file_name)  # training image B

    ## ------------ epoch loop -------------------------
    for epoch in range(1, config.TRAIN.num_epochs + 1):
        epoch_dir = config.TRAIN.out_dir + "/%04d" % epoch
        if os.path.isdir(epoch_dir):
            continue
        cnt = 0

        ## ------------ batch loop -------------------------
        for ind in np.random.permutation(len(train_file_list)):#
            st = time.time()
            cnt += 1

            A_file_name = config.base_dir + config.TRAIN.A_data_dir + '/' + train_file_list[ind]
            if not os.path.isfile(A_file_name) or not os.path.isfile(A_file_name):
                continue
            A_image = helper.read_image(A_file_name)  # training image A


            # may try lr:min(1e-6*np.power(1.1,epoch-1),1e-4 if epoch>100 else 1e-3) in case lr:1e-4 is not good
            feed_dict = {input_A: A_image, input_B: B_image, lr: 1e-4}
            #session run
            eval = fetcher.fetch(feed_dict, [CX_style_loss, CX_content_loss])
            g_loss[ind] = eval[G_loss]

            log = "epoch:%d | cnt:%d | time:%.2f | loss:%.2f || dis_style:%.2f  | dis_content:%.2f " % \
                  (epoch, cnt, time.time() - st, np.mean(g_loss[np.where(g_loss)]), eval[CX_style_loss], eval[CX_content_loss])
            print(log)
         ##------------ end batch loop -------------------

        # save the model
        # we use loop with try and catch to verify that the save was done. when saving on Dropbox it sometimes cause an error.
        for i in range(5):
            try:
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                helper.write_loss_in_txt(g_loss, epoch)
                saver.save(sess, config.TRAIN.out_dir + "/model.ckpt")
            except:
                time.sleep(1)

        ## ------------ validation loop -------------------------
        for ind in range(len(val_file_list)):
            A_file_name_val = config.base_dir + config.VAL.A_data_dir + '/' + val_file_list[ind]
            if not os.path.isfile(A_file_name_val):  # test label
                continue
            A_image_val = helper.read_image(A_file_name_val)  # training image A
            # B_image_val = helper.read_image(B_file_name_val)  # training image A
            output = sess.run(generator_test, feed_dict={input_A_test: A_image_val})
            output = np.concatenate([A_image_val, output, B_image], axis=2)
            helper.save_image(output, config.TRAIN.out_dir + "/%04d/" % epoch + val_file_list[ind].replace('.jpg', '_out.jpg'))



# ---------------------------------------------------
#                      test
# ---------------------------------------------------
if config.TEST.is_test:
    test_file_list = os.listdir(config.base_dir + config.TEST.A_data_dir)
    if not os.path.isdir(config.TEST.out_dir + config.TEST.out_dir_postfix):
        os.makedirs(config.TEST.out_dir + config.TEST.out_dir_postfix)
    time_list = np.zeros(len(test_file_list), dtype=float)
    for ind in range(len(test_file_list)):
        A_file_name_val = config.base_dir + config.TEST.A_data_dir + '/' + test_file_list[ind]
        if not os.path.isfile(A_file_name_val):  # test label
            continue
        A_image_val = helper.read_image(A_file_name_val, fliplr=False)  # training image A
        st = time.time()
        output = sess.run(generator_test, feed_dict={input_A_test: A_image_val})
        et = time.time()
        output = np.concatenate([A_image_val, output], axis=2)#B_image_val
        helper.save_image(output, config.TEST.out_dir + config.TEST.out_dir_postfix + "/" + test_file_list[ind].replace('.jpg', '_out.jpg'))
        time_list[ind] = et - st
        print("test for image #: %d, time: %1.4f" % (ind, et - st))
    print('average time per image: %f' % time_list.mean())
