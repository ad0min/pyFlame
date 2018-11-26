import numpy as np 
from util import batch_iterator

class NeuralNetwork:
    
    def __init__(self, loss_function, validation_data=None):
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss_function 

        self.val_set = None
        # if validation_data:
        #     X, y = validation_data
        #     self.val_set = {"X": X, "y": y}

    def add(self, layer):
        if self.layers:
            # print("shape: ", self.layers[-1].output_shape())
            layer.set_input_shape(shape=self.layers[-1].output_shape())

        if hasattr(layer, 'initialize'):
            layer.initialize()

        self.layers.append(layer)

    def fit(self, X, y, n_epochs, batch_size ):
        for _ in range(n_epochs):
            batch_error = []
            for X_batch, y_batch in batch_iterator(X, y, batch_size = batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)
            self.errors["training"].append(np.mean(batch_error))
            # print(np.mean(batch_error))
        return self.errors["training"]
    
    def train_on_batch(self, X, y):
        """ Single gradient update over one batch of samples """
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Backpropagate. Update weights
        self._backward_pass(loss_grad=loss_grad)

        return loss, acc

    def test_on_batch(self, X, y):
        """ Evaluates the model over a single batch of samples """
        y_pred = self._forward_pass(X, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)

        return loss, acc
    def _forward_pass(self, X, training = True):
        layer_input = X
        for layer in self.layers:
            layer_input = layer.forward_pass(layer_input, training)
        return layer_input

    def  _backward_pass(self, loss_grad):
        for layer in  reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)

    def predict(self, X):
        return self._forward_pass(X, training= False)