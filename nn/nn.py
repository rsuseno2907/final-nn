# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
from math import ceil

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        
        W = np.asarray(W_curr)
        A_prev = np.asarray(A_prev)
        b = np.asarray(b_curr)

        # SHAPE VALIDATION (for debug purposes)
        # if W.ndim != 2:
        #     raise ValueError("W_curr must be 2D (n_curr, n_prev).")
        # if A_prev.ndim != 2:
        #     raise ValueError("A_prev must be 2D (n_prev, m).")
        # n_curr, n_prev = W.shape
        # if A_prev.shape[0] != n_prev:
        #     raise ValueError(f"Shape mismatch: W.shape[1] ({n_prev}) != A_prev.shape[0] ({A_prev.shape[0]}).")
        # if b.ndim == 1:
        #     b = b.reshape((n_curr, 1))
        # elif b.shape == (n_curr,):
        #     b = b.reshape((n_curr, 1))
        # elif b.shape != (n_curr, 1):
        #     raise ValueError(f"b_curr must be shape (n_curr,1) or (n_curr,), got {b.shape}.")

        # calculate z = weights*(activation from previous layer) + bias
        Z_curr = W @ A_prev + b  # shape (n_curr, m)

        # apply activation function
        act = activation.lower()
        if act == "relu":
            A_curr = self._relu(Z_curr)
        elif act == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
        elif act == "linear":
            A_curr = Z_curr
        else:
            raise ValueError("Only sigmoid, relu, and linear activation functions are supported!")
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        A_prev = np.asarray(X)
        # basic validation
        if A_prev.ndim != 2:
            raise ValueError(f"X must be 2D with shape (n_features, m). Got shape {A_prev.shape}.")

        # prepare caches container
        self.caches = []

        # number of layers inferred from architecture
        n_layers = len(self.arch)

        # important loop! this goes through every layer in the network and calls single_forward function
        for l in range(1, n_layers + 1):
            W = np.asarray(self._param_dict[f"W{l}"])
            b = np.asarray(self._param_dict[f"b{l}"])
            activation = self.arch[l-1].get("activation", "linear")

            # compute forward for single layer
            A_curr, Z_curr = self._single_forward(W, b, A_prev, activation)

            # store cache for backprop
            cache = {
                "W": W,
                "b": b,
                "Z": Z_curr,
                "A_prev": A_prev,
                "activation": activation
            }
            self.caches.append(cache)

            # prepare for next layer
            A_prev = A_curr

        # A_prev now holds the activation of the last layer
        return A_prev

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # ensure numpy arrays
        W = np.asarray(W_curr)
        Z = np.asarray(Z_curr)
        Aprev = np.asarray(A_prev)
        dA = np.asarray(dA_curr)

        # determine m (number of examples)
        if Aprev.ndim != 2:
            raise ValueError("A_prev must be 2D with shape (n_prev, m).")
        m = Aprev.shape[1]

        # compute dZ depending on activation
        act = activation_curr.lower()
        if act == "relu":
            dZ = self._relu_backprop(dA, Z)
        elif act == "sigmoid":
            dZ = self._sigmoid_backprop(dA, Z)
        elif act == "linear":
            dZ = dA
        else:
            raise ValueError("Only sigmoid, relu, and linear activation functions are supported!")

        # gradients
        dW_curr = (1.0 / m) * (dZ @ Aprev.T)        # shape (n_curr, n_prev)
        db_curr = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)  # shape (n_curr, 1)
        dA_prev = W.T @ dZ                          # shape (n_prev, m)

        return dA_prev, dW_curr, db_curr

    def backprop(self, X_batch: ArrayLike, y_batch: ArrayLike):
        """
        Compute gradients for one forward/backward pass on provided batch X_batch,y_batch.

        Returns:
            grads: dict with keys "dW1","db1",... for each layer
            loss: float value of loss on this batch
        """
        X = np.asarray(X_batch)
        y = np.asarray(y_batch)

        # forward pass (will populate self.caches)
        A_L = self.forward(X)

        # compute loss and dA_L depending on configured loss
        lf = self._loss_func.lower() if isinstance(self._loss_func, str) else self._loss_func
        if lf in ("binary_cross_entropy", "bce"):
            loss = self._binary_cross_entropy(y, A_L)
            dA = self._binary_cross_entropy_backprop(y, A_L)
        elif lf in ("mean_squared_error", "mse"):
            loss = self._mean_squared_error(y, A_L)
            dA = self._mean_squared_error_backprop(y, A_L)
        else:
            raise ValueError(f"Unsupported loss function '{self._loss_func}'")

        grads = {}
        dA_curr = dA
        n_layers = len(self.arch)

        # iterate layers backward
        for l in range(n_layers, 0, -1):
            cache = self.caches[l - 1]
            W_curr = cache["W"]
            b_curr = cache["b"]
            Z_curr = cache["Z"]
            A_prev = cache["A_prev"]
            activation = cache.get("activation", "linear")

            dA_prev, dW_curr, db_curr = self._single_backprop(
                W_curr=W_curr,
                b_curr=b_curr,
                Z_curr=Z_curr,
                A_prev=A_prev,
                dA_curr=dA_curr,
                activation_curr=activation
            )

            grads[f"dW{l}"] = dW_curr
            grads[f"db{l}"] = db_curr

            dA_curr = dA_prev

        return grads, loss

    def _update_params(self, grads: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        # Adam hyperparameters (standard defaults)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = getattr(self, "_lr", None)
        if lr is None:
            raise AttributeError("Learning rate not found: expected self._lr to be set.")

        # number of layers determined from architecture
        n_layers = len(self.arch)

        # initialize Adam moment accumulators on first call
        if not hasattr(self, "_adam_m"):
            self._adam_m = {}  # first moment
        if not hasattr(self, "_adam_v"):
            self._adam_v = {}  # second moment
        if not hasattr(self, "_adam_t"):
            self._adam_t = 0   # timestep

        m_dict = self._adam_m
        v_dict = self._adam_v

        # increment timestep
        self._adam_t += 1
        t = self._adam_t

        for l in range(1, n_layers + 1):
            W_key = f"W{l}"
            b_key = f"b{l}"
            dW_key = f"dW{l}"
            db_key = f"db{l}"

            # defensive checks with informative errors
            if dW_key not in grads:
                raise KeyError(f"Gradient dict missing '{dW_key}' for layer {l}. Got keys: {list(grads.keys())}")
            if db_key not in grads:
                raise KeyError(f"Gradient dict missing '{db_key}' for layer {l}. Got keys: {list(grads.keys())}")

            dW = grads[dW_key]
            db = grads[db_key]

            # initialize moments if necessary (shape matched to gradients)
            mW = m_dict.get(f"mW{l}", np.zeros_like(dW))
            mb = m_dict.get(f"mb{l}", np.zeros_like(db))
            vW = v_dict.get(f"vW{l}", np.zeros_like(dW))
            vb = v_dict.get(f"vb{l}", np.zeros_like(db))

            # update biased first moment estimate
            mW = beta1 * mW + (1 - beta1) * dW
            mb = beta1 * mb + (1 - beta1) * db

            # update biased second raw moment estimate
            vW = beta2 * vW + (1 - beta2) * (dW * dW)
            vb = beta2 * vb + (1 - beta2) * (db * db)

            # compute bias-corrected first and second moments
            mW_hat = mW / (1 - beta1**t)
            mb_hat = mb / (1 - beta1**t)
            vW_hat = vW / (1 - beta2**t)
            vb_hat = vb / (1 - beta2**t)

            # parameter updates
            self._param_dict[W_key] = self._param_dict[W_key] - lr * (mW_hat / (np.sqrt(vW_hat) + eps))
            self._param_dict[b_key] = self._param_dict[b_key] - lr * (mb_hat / (np.sqrt(vb_hat) + eps))

            # store updated moments back to the instance dicts
            m_dict[f"mW{l}"] = mW
            m_dict[f"mb{l}"] = mb
            v_dict[f"vW{l}"] = vW
            v_dict[f"vb{l}"] = vb

        # save back (probably unnecessary because dicts are mutable, but keep for clarity)
        self._adam_m = m_dict
        self._adam_v = v_dict

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set (for autoencoder, y_train == X_train).
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set (for autoencoder, y_val == X_val).

        Returns:
            history: dict with keys "train_loss" and optional "val_loss"
        """

        # convert inputs to arrays
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val) if X_val is not None else None
        y_val = np.asarray(y_val) if y_val is not None else None

        # allow override of internal defaults
        epochs = self._epochs
        batch_size = self._batch_size

        # create shorthand variables so the rest of the code can use X / y
        X = X_train
        y = y_train

        # basic shape checks (expect features x examples)
        if X.ndim != 2:
            raise ValueError("X must be 2D (features, examples)")
        if y.ndim not in (1, 2):
            raise ValueError("y must be 1D or 2D with examples as columns")

        # convert y shape to (out_features, m) if necessary
        if y.ndim == 1:
            y = y.reshape((1, y.shape[0]))
        if y.shape[1] != X.shape[1]:
            # try transpose if user passed (examples, features)
            if X.shape[0] != self.arch[0]["input_dim"] and X.shape[1] == self.arch[0]["input_dim"]:
                X = X.T
            if y.shape[0] == X.shape[0] and y.shape[1] != X.shape[1]:
                y = y.T
        m = X.shape[1]

        history = {"train_loss": [], "val_loss": [] if X_val is not None else None}

        # RNG using your class seed attribute
        rng = np.random.RandomState(getattr(self, "_seed", None))

        # verbosity: use self._verbose if defined, otherwise default to 1
        verbose = getattr(self, "_verbose", 1)

        for epoch in range(1, epochs + 1):  # for each epoch...
            # shuffle indices to switch up the order of examples (prevent bias learned by model)
            perm = rng.permutation(m)
            X_shuf = X[:, perm]
            y_shuf = y[:, perm]

            # minibatch loop
            n_batches = ceil(m / batch_size)
            epoch_losses = []
            for i in range(0, m, batch_size):  # for each batch...
                X_batch = X_shuf[:, i : i + batch_size]
                y_batch = y_shuf[:, i : i + batch_size]

                grads, batch_loss = self.backprop(X_batch, y_batch)
                self._update_params(grads)

                epoch_losses.append(batch_loss)
                # ... perform forward and backprop followed by update_params

            # end epoch: compute metrics on full training set
            # forward over full set to compute final train loss for epoch
            A_train = self.forward(X)
            if getattr(self, "_loss_func", "mean_squared_error").lower() in ("binary_cross_entropy", "bce"):
                train_loss = self._binary_cross_entropy(y, A_train)
            else:
                train_loss = self._mean_squared_error(y, A_train)

            val_loss = None
            if X_val is not None and y_val is not None:
                X_val_arr = np.asarray(X_val)
                y_val_arr = np.asarray(y_val)
                # ensure shapes (features, examples)
                if X_val_arr.ndim != 2:
                    raise ValueError("X_val must be 2D (features, examples)")
                if y_val_arr.ndim == 1:
                    y_val_arr = y_val_arr.reshape((1, y_val_arr.shape[0]))
                A_val = self.forward(X_val_arr)
                if getattr(self, "_loss_func", "mean_squared_error").lower() in ("binary_cross_entropy", "bce"):
                    val_loss = self._binary_cross_entropy(y_val_arr, A_val)
                else:
                    val_loss = self._mean_squared_error(y_val_arr, A_val)
                history["val_loss"].append(val_loss)

            history["train_loss"].append(train_loss)

            if verbose:
                if val_loss is not None:
                    print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
                else:
                    print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.6f}")

        # tidy history: remove val list if not used
        if history["val_loss"] is None:
            history.pop("val_loss", None)

        return history


    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        X = np.asarray(X)
        return self.forward(X)

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        A = self._sigmoid(Z)
        dZ = dA * A * (1 - A)
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # relu returns x if x>0 and returns 0 if x<=0
        return np.maximum(0,Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # turn entries in dA (from previous layer) 
        dZ = np.array(dA, copy=True)  # copy to avoid modifying original
        dZ[Z <= 0] = 0
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        bce = -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
        return bce

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # clip y_hat in case it's equal to 0 and 1 which otherwise would be a problem
        eps = 1e-8
        y_hat = np.clip(y_hat, eps, 1 - eps)
        
        # m = y.shape[0] if y.ndim == 1 else y.shape[1]
        m = y.shape[1]
        dA = (-y / y_hat + (1 - y) / (1 - y_hat)) / m
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        mse = np.square(np.subtract(y,y_hat)).mean()
        return mse

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        N = y.size
        return (2 / N) * (y_hat - y)