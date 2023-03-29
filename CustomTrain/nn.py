import numpy as np
import scipy


class Linear:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, inp):
        """
        Implement the linear part of a layer's forward propagation.

        Args:
            inp : activations from previous layer (or input data)

        Returns:

            z  : the input of the activation function, also called pre-activation parameter
        """
        self.inp = inp
        self.z = (inp @ self.w) + self.b
        return self.z

    def backward(self, grads):
        """
        Implement the linear portion of backward propagation for a single layer.

        Args:
            grads :  Gradient of the cost with respect to the linear output.
                     or the accumulated gradients from the prev layers.
                     This is used for the chain rule to compute the gradients.
        Returns:
            da : Gradient of cost wrt to the activation of the previous layer or the input of the
                 current layer.
            dw : Gradient of the cost with respect to W
            db : Gradient of the cost with respect to b
        """
        m = self.inp.shape[1]
        # gradient of loss wrt to the weights
        dw = 1 / m * (self.inp.T @ grads)
        # gradient of the loss wrt to the bias
        db = 1 / m * np.sum(grads, axis=0, keepdims=True)
        # gradient of the loss wrt to the input of the linear layer
        # this is used to continue the chain rule
        da_prev = grads @ self.w.T
        return da_prev, dw, db


class SparseLinear:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, inp):
        """
        Implement the linear part of a layer's forward propagation.

        Args:
            inp : activations from previous layer (or input data)

        Returns:

            z  : the input of the activation function, also called pre-activation parameter
        """
        self.inp = inp
        self.z = scipy.sparse.csr_matrix.dot(inp, self.w) + self.b
        return self.z

    def backward(self, grads):
        """
        Implement the linear portion of backward propagation for a single layer.

        Args:
            grads :  Gradient of the cost with respect to the linear output.
                     or the accumulated gradients from the prev layers.
                     This is used for the chain rule to compute the gradients.
        Returns:
            da : Gradient of cost wrt to the activation of the previous layer or the input of the
                 current layer.
            dw : Gradient of the cost with respect to W
            db : Gradient of the cost with respect to b
        """
        m = self.inp.shape[1]
        # gradient of loss wrt to the weights
        dw = 1 / m * scipy.sparse.csr_matrix.dot(self.inp.T, grads)
        # gradient of the loss wrt to the bias
        db = 1 / m * np.sum(grads, axis=0, keepdims=True)
        # gradient of the loss wrt to the input of the linear layer
        # this is used to continue the chain rule
        da_prev = grads @ self.w.T
        return da_prev, dw, db


class Relu:
    def forward(self, inp):
        """
        Implement the RELU function.

        Args:
            inp : Output of the linear layer, of any shape

        Returns:
            a  : Post-activation parameter, of the same shape as Z
        """
        self.inp = inp
        self.output = np.maximum(0, self.inp)
        return self.output

    def backward(self, grads):
        """
        Implement the backward propagation for a single RELU unit.

        Ars:
            grads : gradients of the loss wrt to the activation output

        Returns:
            dz : Gradient of the loss with respect to the input of the activation
        """
        dz = np.array(grads, copy=True)
        dz[self.inp <= 0] = 0
        return dz


class CCELoss:
    def forward(self, logits, label):
        """
        Implement the CrossEntropy loss function.

        Args:
            pred   : predicted labels from the neural network
            target : true "label" labels
        Returns:
            loss   : cross-entropy loss
        """
        self.yhat = logits
        self.y = label
        # m = self.y.shape[0]
        # commpute loss
        # label should be scaled
        max_logit = np.max(logits, axis=1, keepdims=True)
        # softmax stable exponential
        pred_exp = np.exp(logits - max_logit)
        exp_sum = np.sum(pred_exp, axis=1, keepdims=True)
        self.softmax = pred_exp / exp_sum
        log_sm = logits - max_logit - np.log(exp_sum)
        # compute loss per data in batch
        sample_losses = -np.sum(log_sm * label, axis=1)
        self.output = np.mean(sample_losses)
        return self.output

    def backward(self):
        """
        Computes the gradinets of the loss_fn wrt to the predicted labels

        Returns:
         da : derivative of loss_fn wrt to the predicted labels
        """
        # derivative of loss_fn with respect to a [predicted labels]
        da = np.where(self.y > 0, self.y * (self.softmax - 1) + (1 - self.y) * self.softmax, self.softmax)
        return da


class Model:
    def __init__(self, w1, b1, w2, b2, learning_rate):
        """
        A simple neural network model
        The `forward` method computes the forward propagation step of the model
        The `backward` method computes the backward step propagation of the model
        The `update_step` method updates the parameters of the model
        """
        self.lin1 = SparseLinear(w1, b1)  # 1st linear layer
        self.relu1 = Relu()  # 1st activation layer
        self.lin2 = Linear(w2, b2)  # 2nd linear layer
        self.loss_fn = CCELoss()  # loss_fn

        # learning_rate to update model parameters
        self.lr = learning_rate
        # stores the loss at each iteration
        self.losses = []

    def forward(self, inp, targ=None, calc_loss=True):
        """
        Computs the forward step for out model Additionally
        it also returns the loss [Optional] and the predictions
        of the model.

        Args:
            inp       : the training set.
            calc_loss : wether to calculate loss of the model if False only predictions
                        are calculated.
            targ      : the original targets to the training set.

        Note: to calculate the `loss` the `targ` must be given

        Returns:
            pred : outputs of the 3rd activation layer.
            loss : [Optional] loss the model , if the `targ` is given.
        """
        pred = self.lin2.forward(self.relu1.forward(self.lin1.forward(inp)))

        if calc_loss:
            assert targ is not None, "to calculate loss targets must be given"
            loss = self.loss_fn.forward(pred, targ)
            # appending the loss of the current iteration
            self.losses.append(loss)
            return loss, pred
        else:
            return pred

    #'''
    def _assert_shapes(self):
        """
        Checks the shape of the parameters and the gradients of the model
        """
        assert self.lin1.w.shape == self.dw1.shape
        assert self.lin2.w.shape == self.dw2.shape

        assert self.lin1.b.shape == self.db1.shape
        assert self.lin2.b.shape == self.db2.shape
    #'''

    def backward(self):
        """
        Computes the backward step
        and return the gradients of the parameters with the loss
        """
        dz2 = self.loss_fn.backward()
        da2, self.dw2, self.db2 = self.lin2.backward(dz2)

        dz1 = self.relu1.backward(da2)
        _, self.dw1, self.db1 = self.lin1.backward(dz1)

        self._assert_shapes()

        self.dws = [self.dw1, self.dw2]
        self.dbs = [self.db1, self.db2]

    def update(self):
        """
        Performs the update step
        """
        self.lin1.w -= self.lr * self.dws[0]
        self.lin2.w -= self.lr * self.dws[1]

        self.lin1.b -= self.lr * self.dbs[0]
        self.lin2.b -= self.lr * self.dbs[1]
