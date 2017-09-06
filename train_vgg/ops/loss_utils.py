import numpy as np
import tensorflow as tf
import tf_fun
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from ops.tf_fun import fine_tune_prepare_layers


def wd_loss(
        loss,
        trainables,
        config):

    _, l2_wd_layers = fine_tune_prepare_layers(
        tf.trainable_variables(), config.wd_layers)
    l2_wd_layers = [
        x for x in l2_wd_layers if 'biases' not in x.name]
    loss += (
        config.wd_penalty * tf.add_n(
            [tf.nn.l2_loss(x) for x in l2_wd_layers]))
    return loss


def finetune_learning(
        loss,
        trainables,
        config):
    if config.fine_tune_layers is not None:
        other_opt_vars, ft_opt_vars = fine_tune_prepare_layers(
            tf.trainable_variables(), config.fine_tune_layers)
        if config.optimizer == 'adam':
            train_op, gvs = ft_optimizer_list(
                loss, [other_opt_vars, ft_opt_vars],
                tf.train.AdamOptimizer,
                [config.hold_lr, config.new_lr],
                grad_clip=config.grad_clip)
        elif config.optimizer == 'sgd':
            train_op, gvs = ft_optimizer_list(
                loss, [other_opt_vars, ft_opt_vars],
                tf.train.GradientDescentOptimizer,
                [config.hold_lr, config.new_lr],
                grad_clip=config.grad_clip)
        else:
            raise RuntimeError('Your specified optimizer is not yet implemented.')
    else:
        if config.optimizer == 'adam':
            train_op = tf.train.AdamOptimizer(
                config.hold_lr).minimize(loss)
        elif config.optimizer == 'sgd':
            train_op = tf.train.GradientDescentOptimizer(
                config.hold_lr).minimize(loss)
        else:
            raise RuntimeError('Your specified optimizer is not yet implemented.')
    return train_op


def inception_finetune_learning(loss, cmcfg, incfg):
    '''
    Same as above, but for inception networks.
    cmcfg is a clickMeConfig, incfg is an InceptionConfig.
    '''
    slim = tf.contrib.slim
    opts = {'adam': tf.train.AdamOptimizer,
            'sgd': tf.train.GradientDescentOptimizer}
    opt_fn = opts.get(cmcfg.optimizer)
    if not opt_fn:
        raise ValueError('%s needs to be added to `inception_finetune_learning(...)'
                         % cmcfg.optimizer)
    if incfg.trainable_scopes:
        vars_to_finetune, other_vars = [], []
        for var in slim.get_model_variables():
            if any(map(lambda t: var.op.name.startswith(t), incfg.trainable_scopes)):
                vars_to_finetune.append(var)
            else:
                other_vars.append(var)
        train_op, gvs = ft_optimizer_list(loss, [other_vars, vars_to_finetune],
                                          opt_fn, [cmcfg.hold_lr, cmcfg.new_lr],
                                          grad_clip=cmcfg.grad_clip)
        return train_op
    else:
        return opt_fn(cmcfg.hold_lr).minimize(loss)
        



def l2_loss(y, yhat):
    return tf.nn.l2_loss(yhat - y)


def huber_loss(y, yhat, k=4.):
    delta = yhat - y
    abs_delta = tf.abs(delta)
    cond_delta = delta < k
    sq_delta = tf.square(delta)
    loss = tf.reduce_mean(
        tf.where(
            cond_delta,
            0.5 * sq_delta,
            k * abs_delta - 0.5 * (k ** 2))
        )
    return loss


def adj_l2(y, yhat, eps=1e-2):
    return tf.reduce_mean(tf.pow((yhat - y / (1 - y + eps)), 2))


def loss_switch(ltype):
    if ltype == 'l2':
        return l2_loss
    elif ltype == 'adj_l2':
        return adj_l2
    elif ltype == 'huber':
        return huber_loss
    else:
        raise RuntimeError(
            'Cannot understand what kind of mapping loss you want to use.')


def ft_optimizer(cost, other_opt_vars, ft_opt_vars, optimizer, lr_1, lr_2):
    """Efficient optimization for fine tuning a net."""
    op1 = optimizer(lr_1).minimize(cost, var_list=other_opt_vars)
    op2 = optimizer(lr_2).minimize(cost, var_list=ft_opt_vars)
    return tf.group(op1, op2)


def ft_optimizer_list(cost, opt_vars, optimizer, lrs, grad_clip=False):
    """Efficient optimization for fine tuning a net."""
    ops = []
    gvs = []
    for v, l in zip(opt_vars, lrs):
        if grad_clip:
            optim = optimizer(l)
            gvs = optim.compute_gradients(cost, var_list=v)
            capped_gvs = [
                (tf.clip_by_norm(grad, 10.), var)
                if grad is not None else (grad, var) for grad, var in gvs]
            ops.append(optim.apply_gradients(capped_gvs))
        else:
            ops.append(optimizer(l).minimize(cost, var_list=v))
    return tf.group(*ops), gvs


def add_decay(loss, model, attention_layers, norm_type='l2', beta=0.01):
    if norm_type == 'l1':
        decay_norm = l1_norm
    elif norm_type == 'l2':
        decay_norm = l2_norm
        norm_type = channel_l2_normalize
    for l in attention_layers:
        loss += decay_norm(model[l])
    return loss * beta


def l2_distance(x, y, use_sqrt=True):
    """Distance between net and clickme attention."""
    if use_sqrt:
        return tf.reduce_mean(
            tf.sqrt(
                tf.reduce_mean(tf.pow(x - y, 2), axis=[1, 2])))
    else:
        return tf.reduce_mean(tf.pow(x - y, 2))


def l1_loss(x):
    return tf.reduce_sum(tf.abs(x))


def l2_loss(lrp_maps, train_heatmaps):
    """Per channel L2 normalization followed by combining the L2 loss
    with an existing loss."""
    return l2_distance(
               channel_l2_normalize(lrp_maps),
               channel_l2_normalize(train_heatmaps), use_sqrt=False)


def cosine_distance(x, y):
    return tf.constant(1.) - tf.reduce_sum(tf.reduce_sum(
        tf.multiply(x, y), axis=[1, 2]))


def l1_norm(x, reduction_indices=[0, 1, 2, 3]):
    return tf.reduce_sum(tf.abs(x), axis=reduction_indices)


def l2_norm(x, reduction_indices=[0, 1, 2, 3]):
    return tf.sqrt(
        tf.reduce_sum(tf.pow(x, 2), axis=reduction_indices))


def channel_l2_normalize(x, eps=1e-12):
    norm = tf.sqrt(
        tf.reduce_sum(tf.pow(x, 2), axis=[1, 2], keep_dims=True))
    return x / (norm + eps)


def l2_normalize(x, dim=3):
    """L2 normalize across batch dimension."""
    return tf.nn.l2_normalize(x, dim=dim)


def combine(x, fun, layer_name=None, p=2):
    if fun == 'sum_abs':
        comb_x = tf.reduce_sum(
            tf.abs(x), axis=[3], keep_dims=True)
    elif fun == 'sum_p':
        comb_x = tf.reduce_sum(
            tf.pow(tf.abs(x), p), axis=[3], keep_dims=True)
    elif fun == 'max_p':
        comb_x = tf.reduce_max(
            tf.pow(tf.abs(x), p), axis=[3], keep_dims=True)
    elif fun == 'max_abs':
        comb_x = tf.reduce_max(
            tf.abs(x), axis=[3], keep_dims=True)
    elif fun == 'max':
        comb_x = tf.reduce_max(x, axis=[3], keep_dims=True)
    elif fun == 'max_relu':
        comb_x = tf.nn.relu(
            tf.reduce_max(x, axis=[3]), keep_dims=True)
    elif fun == 'mlp':
        filt, biases = get_conv_var(1, x.get_shape()[-1], 1, layer_name)
        comb_x = tf.nn.relu(
            tf.nn.conv2d(x, filt, [1, 1, 1, 1], padding='SAME'))
    elif fun == 'relu_sum':
        comb_x = tf.reduce_sum(
            tf.nn.relu(x), axis=[3], keep_dims=True)
    elif fun == 'relu_sum_p':
        comb_x = tf.reduce_sum(
            tf.nn.relu(x) ** 2, axis=[3], keep_dims=True)
    elif fun == 'relu_sum_inv':
        comb_x = tf.reduce_sum(
            tf.nn.relu(x * -1), axis=[3], keep_dims=True)
    elif fun == 'relu_sum_p_inv':
        comb_x = tf.reduce_sum(
            tf.nn.relu(x * -1) ** 2, axis=[3], keep_dims=True)
    elif fun == 'relu':
        comb_x = tf.nn.relu(x)
    elif fun == 'pass':
        comb_x = x
    else:
        raise Exception
    return comb_x


def get_conv_var(filter_size, in_channels, out_channels, name):
    initial_value = tf.truncated_normal(
        [filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
    filters = get_var(name + '_filters'. initial_value)
    initial_value = tf.truncated_normal([out_channels], .0, .001)
    biases = get_var(initial_value, name, 1, name + '_biases')
    return filters, biases


def get_var(var_name, value):
    return tf.get_variable(name=var_name, initializer=value)


def attention_distance(
        attention, heatmaps, nodes, loss_distance='l2', beta=0.01):
    """Calculate l2 regularized loss at each node between gradient on the first
    pass through the network and a clickme realization map.

    attention can be either activation maps
    or it can be gradient w.r.t. input
    or potentially"""

    if loss_distance == 'l2':
        dist = l2_distance
    elif loss_distance == 'cosine':
        dist = cosine_distance
    att_diff = {}
    for node in nodes:
        # When this is gradient it becomes the mean l2 diff
        # of each grad layer and the heatmap.
        att_diff[node] = beta * dist(
            l2_normalize(attention[node]),
            l2_normalize(
                tf.squeeze(
                    tf_fun.resize(heatmaps, attention[node]))))
    return att_diff, l2_normalize(attention[nodes[0]])


def attention_regularization(
        gradient, heatmaps, nodes, l=0.1):

    pass


def calculate_gradients(loss, model, attention_layers, combine_type):
    # grads = {}
    # for x in attention_layers:
    #     grads[x] = tf.gradients(loss, model[x])[0]
    # return grads
    return {k: combine(
        tf.gradients(loss, model[k])[0], combine_type)
        for k in attention_layers}


def calculate_activations(model, attention_layers, combine_type):
    return {k: combine(model[k], combine_type) for k in attention_layers}


def combine_losses(cat_loss, at_loss, names):
    """ Addes losses together.
    Because losses are held in tuples we have to extract their elements
    then build a new one."""
    for k, v in at_loss.iteritems():
        idx = names.index(k)
        it_cat_loss = cat_loss[idx]
        loss_tensor = it_cat_loss[0]
        var_tensor = it_cat_loss[1]
        loss_tensor += v
        cat_loss[idx] = (loss_tensor, var_tensor)
    return cat_loss


def route_attention_loss(
        cat_att, cat_loss, heatmaps, attention_layers, loss_distance,
        attention_loss):
    if attention_loss == 'distance':
        at_loss, grad_im = attention_distance(
            cat_att, heatmaps,
            attention_layers,
            loss_distance=loss_distance)
    elif attention_loss == 'regularization':
        at_loss, grad_im = attention_regularization(
            cat_loss, heatmaps,
            attention_layers,
            loss_distance=loss_distance)
    else:
        raise Exception
    return at_loss, grad_im


def double_backprop_ft(
        model, loss, heatmaps, optimizer, config, other_opt_vars, ft_opt_vars):
    op_other = optimizer(config.hold_lr)
    op_ft = optimizer(config.new_lr)

    # Figure out which attention nodes are also being finetuned
    other_names, ft_names = tf_fun.fine_tune_names(
        tf.trainable_variables(), config.fine_tune_layers)
    trim_other_names = tf_fun.filter_node_names(other_names)
    trim_ft_names = tf_fun.filter_node_names(ft_names)
    other_attention_layers = tf_fun.list_intersection(
        trim_other_names, config.attention_layers)
    ft_attention_layers = tf_fun.list_intersection(
        trim_ft_names, config.attention_layers)

    # Calculate classification gradients
    cat_loss_other = op_other.compute_gradients(loss, var_list=other_opt_vars)
    cat_loss_ft = op_ft.compute_gradients(loss, var_list=ft_opt_vars)

    # Calculate attention maps for the net
    if config.attention_type == 'gradient':
        cat_att_other = calculate_gradients(
            loss, model, other_attention_layers,
            combine_type=config.combine_type)
        cat_att_ft = calculate_gradients(
            loss, model, ft_attention_layers,
            combine_type=config.combine_type)
    elif config.attention_type == 'activation':
        cat_att_other = calculate_activations(
            model, other_attention_layers,
            combine_type=config.combine_type)
        cat_att_ft = calculate_activations(
            model, ft_attention_layers,
            combine_type=config.combine_type)

    # Calculate attention loss
    at_loss_other = None
    at_loss_ft = None
    grad_im = None
    if len(other_attention_layers) > 0:
        at_loss_other, grad_im = route_attention_loss(
            cat_att_other, cat_loss_other, heatmaps,
            config.attention_layers,
            loss_distance=config.loss_distance,
            attention_loss=config.attention_loss)

    if len(ft_attention_layers) > 0:
        at_loss_ft, grad_im = route_attention_loss(
            cat_att_ft, cat_loss_ft, heatmaps,
            config.attention_layers,
            loss_distance=config.loss_distance,
            attention_loss=config.attention_loss)

    return update_gradient_ft(
            op_other, op_ft, cat_loss_other, cat_loss_ft,
            at_loss_other, at_loss_ft,
            other_opt_vars, ft_opt_vars), grad_im


def update_gradient_ft(
        op_other, op_ft, cat_loss_other, cat_loss_ft,
        at_loss_other, at_loss_ft,
        other_opt_vars, ft_opt_vars):
    """Minimizes loss for a double backprop.
    1st loop is for categorization loss w.r.t. X
    2nd loop is for attention loss w.r.t. clickme realization maps.

    op1, op2 from dbp_first_ft
    cost_1 and cost_2 from dbp_first_ft -> l2_attention
    other_opt_vars, ft_opt_vars from the config.
    """
    # Combine losses from attention with classification
    if at_loss_other is not None:
        other_names = tf_fun.node_names(cat_loss_other)
        cat_loss_other = combine_losses(
            cat_loss_other, at_loss_other, other_names)

    if at_loss_ft is not None:
        ft_names = tf_fun.node_names(cat_loss_ft)
        cat_loss_ft = combine_losses(
            cat_loss_ft, at_loss_ft, ft_names)

    # Manually trigger optimization
    if len(cat_loss_other) == 1 and cat_loss_other[0][0] is None:
        # Training the entire net
        return op_ft.apply_gradients(cat_loss_ft)

    elif len(cat_loss_ft) == 1 and cat_loss_ft[0][0] is None:
        # Not training anything ....
        return op_ft.apply_gradients(cat_loss_other)

    else:
        # Training some layers but not others
        db_op1 = op_other.apply_gradients(cat_loss_other)
        db_op2 = op_ft.apply_gradients(cat_loss_ft)
        return tf.group(db_op1, db_op2)


def softmax_loss(logits, labels, ratio=None):
    if ratio is not None:
        ratios = tf.get_variable(
            name='ratio', initializer=ratio[::-1])[None, :]
        weights_per_label = tf.matmul(
            tf.one_hot(labels, 2), tf.transpose(tf.cast(ratios, tf.float32)))
        return tf.reduce_mean(
            tf.multiply(
                weights_per_label,
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)))
    else:
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def per_im_softmax_loss(logits, labels):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)


#@ops.RegisterGradient("MaxPoolGrad")
def _MaxPoolGradGrad(op, grad):
    dX = tf.zeros_like(op.inputs[0])
    dZ = tf.zeros_like(op.inputs[1])
    dS = tf.nn.avg_pool(
            grad,
            padding=op.get_attr('padding'),
            ksize=op.get_attr('ksize'),
            strides=op.get_attr('strides')) * np.prod(op.get_attr('ksize'))
    return [dX, dZ, dS]


def test_max_pool_grad():
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    shape = [1, 4, 4, 1]

    with tf.Session():
        X = tf.constant(
            np.random.normal(size=shape).astype(np.float32), name="x")
        XS = tf.constant(
            np.random.normal(size=shape).astype(np.float32), name="xs")
        Z = tf.nn.max_pool(X, strides=strides, ksize=ksize, padding=padding)
        S = tf.nn.max_pool(XS, strides=strides, ksize=ksize, padding=padding)
        C = gen_nn_ops._max_pool_grad(X, Z, S, ksize, strides, padding)
        err = tf.test.compute_gradient_error(X, shape, C, shape)

    assert err <= 1e-4
    print '-' * 60
    print 'Passed maxPoolGradGrad test!'
    print '-' * 60


if __name__ == '__main__':
    test_max_pool_grad()
