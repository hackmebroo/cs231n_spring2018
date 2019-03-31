import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    # 第i个样本对应某个标签的分数这里scores是一个向量，每个元素表示
    # 若该样本分类为该元素的下标，得分为该元素的值[1,4,6,3,2]第i个样本
    # 被分类为1,2,3,4,5得分分别为1,4,6,3,2
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # 设置delta等于1表示希望正确分类的分数至少比错误分类的分数大1
      if margin > 0:
        loss += margin
        dW[:,y[i]] += -X[i,:].T
        dW[:,j] += X[i,:].T
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  scores = X.dot(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores_correct = scores[np.arange(num_train),y]
  # 这里生成的scores_correct矩阵元素是每个样本的分类正确的分数
  # scores里面存的是每个样本分到某个类得的分数
  scores_correct = np.reshape(scores_correct,(num_train,-1))
  # scores_correct矩阵变成shape（num_train,1）
  margins = scores - scores_correct + 1
  # scores每一列是每个样本分到某一类的分数，每一列每个元素减去该样本分类正确
  # 得到的分数再加1比如[3.2,5.1,-1.7]分类正确为3.2
  margins = np.maximum(0,margins)
  # 经过处理后得到[1,2.9，0]
  margins[np.arange(num_train),y] = 0
  # 分类正确置0得到[0,2.9,0]即loss为2.9
  loss += np.sum(margins) / num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins[margins > 0] = 1
  row_sum = np.sum(margins, axis = 1)
  margins[np.arange(num_train),y] = -row_sum
  dW += np.dot(X.T,margins)/num_train + reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
