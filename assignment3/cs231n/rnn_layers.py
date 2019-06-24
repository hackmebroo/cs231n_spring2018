from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    temp1 = x.dot(Wx)
    temp2 = prev_h.dot(Wh)
    temp = temp1 + temp2 + b
    next_h = np.tanh(temp)
    cache = (x,prev_h,Wx,Wh,temp)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x,prev_h,Wx,Wh,temp = cache
    N,H = dnext_h.shape[0],dnext_h.shape[1]
    temp_all = np.ones((N,H)) - np.square(np.tanh(temp))
    delta = temp_all * dnext_h      #先对整个tanh求导
    dx = delta.dot(Wx.T)
    dWx = (x.T).dot(delta)
    dprev_h = delta.dot(Wh.T)
    dWh = (prev_h.T).dot(delta)
    db = (np.sum(delta,axis=0)).T
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N,T,D = x.shape
    H = h0.shape[1]
    prev_h = h0
    h1 = np.empty([N,T,H])
    h2 = np.empty([N,T,H])
    h3 = np.empty([N,T,H])
    for i in range(0,T):
        h,cache = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)
        h1[:,i,:] = prev_h
        prev_h = h
        h2[:,i,:] = h
        h3[:,i,:] = cache[4]
    cache = (x,h1,Wx,Wh,h3)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h2, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N,T,H = dh.shape
    x = cache[0]
    D = x.shape[2]
    dWx = np.zeros((D,H))
    dWh = np.zeros((H,H))
    db = np.zeros(H)
    dout = dh
    dx = np.empty([N,T,D])
    dh0 = np.empty([N,T,H])
    dh_now = np.zeros((N,H))
    for j in range(0,T):
        i = T-1-j
        dh_now = dh_now + dout[:,i,:]
        cache_t = (cache[0][:,i,:],cache[1][:,i,:],cache[2],cache[3],cache[4][:,i,:])
        dx_temp,dprev_h_temp,dWx_temp,dWh_temp,db_temp = rnn_step_backward(dh_now,cache_t)
        dh_now = dprev_h_temp
        dx[:,i,:] = dx_temp
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp
        # 因为每一层都使用了共享参数Wx,Wh,b，所以这三个的梯度为所有层的和
    dh0 = dh_now
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    out = W[x,:]
    # 总共V个词，每个词用一个1*D向量表示
    # x为N个样本，每个样本序列长度为T，序列中每个元素为1*D向量
    # 该函数作用为，将输入的x进行编码。
    cache = (W,x)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    W,x = cache
    dW = np.zeros_like(W)
    np.add.at(dW,x,dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N,H = prev_h.shape
    A = x.dot(Wx) + prev_h.dot(Wh) + b
    ai = A[:,:H]
    af = A[:,H:2*H]
    ao = A[:,2*H:3*H]
    ag = A[:,3*H:4*H]
    
    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)
    
    next_c =np.multiply(f,prev_c) + np.multiply(i,g)
    next_h = np.multiply(o,np.tanh(next_c))
    
    cache = (x,prev_h,prev_c,i,f,o,g,Wx,Wh,next_c,A)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    N,H = dnext_h.shape
    prev_x,prev_h,prev_c,i,f,o,g,Wx,Wh,next_c,A = cache
    ai = A[:,:H]
    af = A[:,H:2*H]
    ao = A[:,2*H:3*H]
    ag = A[:,3*H:4*H]
    
    # 到ct-1的梯度
    dc_2 = np.multiply(dnext_c,f)
    temp = np.multiply(dnext_h,o)
    temp1 = np.ones_like(next_c) - np.square(np.tanh(next_c))
    temp2 = np.multiply(temp1,f)
    dc_1 = np.multiply(temp,temp2)
    dprev_c = dc_1 + dc_2
    #dE/dh * dh/dc
    dE_dc = np.multiply(temp,temp1) + dnext_c
    
    #计算di、df、do、dg
    di = np.multiply(dE_dc,g)
    dg = np.multiply(dE_dc,i)
    df = np.multiply(dE_dc,prev_c)
    do = np.multiply(dnext_h,np.tanh(next_c))
    
    #到ai,af,ao,ag
    dao = np.multiply(do,np.multiply(o,(np.ones_like(o) - o)))
    daf = np.multiply(df,np.multiply(f,(np.ones_like(f) - f)))
    dai = np.multiply(di,np.multiply(i,(np.ones_like(i) - i)))
    dtanhg = np.ones_like(ag) - np.square(np.tanh(ag))
    dag = np.multiply(dg,dtanhg)
    
    #计算各参数梯度
    dall = np.concatenate((dai,daf,dao,dag),axis=1)
    dx = dall.dot(Wx.T)
    dprev_h = dall.dot(Wh.T)
    dWx = prev_x.T.dot(dall)
    dWh = prev_h.T.dot(dall)
    db = np.sum(dall,axis=0).T
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N,T,D = x.shape
    H = h0.shape[1]
    prev_h = h0
    
    h = np.empty([N,T,H])
    h2 = np.empty([N,T,H])
    h3 = np.empty([N,T,H])
    h4 = np.empty([N,T,H])
    I = np.empty([N,T,H])
    f = np.empty([N,T,H])
    o = np.empty([N,T,H])
    g = np.empty([N,T,H])
    nc = np.empty([N,T,H])
    A = np.empty([N,T,4*H])
    prev_c = np.zeros_like(prev_h)
    for i in range(0,T):
        h3[:,i,:] = prev_h
        h4[:,i,:] = prev_c
        next_h,next_c,cache_temp = lstm_step_forward(x[:,i,:],prev_h,prev_c,Wx,Wh,b)
        prev_h = next_h
        prev_c = next_c
        h2[:,i,:] = prev_h
        I[:,i,:] = cache_temp[3]
        # cache = (x,prev_h,prev_c,i,f,o,g,Wx,Wh,next_c,A)
        f[:,i,:] = cache_temp[4]
        o[:,i,:] = cache_temp[5]
        g[:,i,:] = cache_temp[6]
#         print(cache_temp[9].shape)
        nc[:,i,:] = cache_temp[9]
#         print(cache_temp[10].shape)
        A[:,i,:] = cache_temp[10]
    
    h = h2
    cache = (x,h3,h4,I,f,o,g,Wx,Wh,nc,A)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
#     cache = (x,h3,h4,i,f,o,g,Wx,Wh,next_c,A)
    x = cache[0]
    N,T,D = x.shape
    H = dh.shape[2]
    
    dWx = np.zeros((D,4*H))
    dWh = np.zeros((H,4*H))
    db = np.zeros(4*H)
    dout = dh
    dx = np.empty([N,T,D])
    dh_now = np.zeros((N,H))
    c_now = np.zeros((N,H))
    for k in range(0,T):
        i = T-1-k
        dh_now = dh_now + dout[:,i,:]
#         print(cache[9].shape)
#         print(cache[10].shape)
        cache_T = (cache[0][:,i,:],cache[1][:,i,:],cache[2][:,i,:],cache[3][:,i,:],
               cache[4][:,i,:],cache[5][:,i,:],cache[6][:,i,:],cache[7],cache[8],
                cache[9][:,i,:],cache[10][:,i,:])
        
        dx_temp, dprev_h, dprev_c, dWx_temp, dWh_temp, db_temp = lstm_step_backward(
        dh_now,c_now,cache_T)
        dh_now = dprev_h
        c_now = dprev_c
        dx[:,i,:] = dx_temp
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp
        
    dh0 = dh_now
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
