# test_nn_additional.py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nn.preprocess import one_hot_encode_seqs, sample_seqs
from nn.io import read_text_file, read_fasta_file
from nn.nn import NeuralNetwork

def test_single_forward():
    model = make_simple_model()
    W = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]) 
    b = np.array([[0.1], [-0.2]])
    A_prev = np.array([[1.0, 2.0],
                       [0.0, -1.0],
                       [2.0,  1.0]])
    # linear activation: A_curr should equal Z
    A_curr, Z_curr = model._single_forward(W_curr=W, b_curr=b, A_prev=A_prev, activation="linear")
    Z_expected = W @ A_prev + b
    assert A_curr.shape == Z_expected.shape
    assert Z_curr.shape == Z_expected.shape
    assert_allclose(Z_curr, Z_expected, atol=1e-8)
    assert_allclose(A_curr, Z_expected, atol=1e-8)

    # sigmoid activation: verify shape and value range
    A_sig, Z_sig = model._single_forward(W_curr=W, b_curr=b, A_prev=A_prev, activation="sigmoid")
    assert A_sig.shape == Z_sig.shape
    assert np.all(A_sig >= 0) and np.all(A_sig <= 1)


def test_forward():
    model = make_simple_model()
    # random input features x examples (shape input_dim x m)
    X = np.array([[0.1, 0.2, 0.3]]).T  # shape (3,1) -> we will transpose to (3,m)
    # Ensure shape (n_features, m_examples)
    X = np.random.RandomState(0).rand(3, 5)
    A_out = model.forward(X)
    # output layer is 1 neuron -> A_out should be shape (1, m)
    assert isinstance(A_out, np.ndarray) or hasattr(A_out, "shape")
    A_out = np.asarray(A_out)
    assert A_out.shape == (1, X.shape[1])
    # caches should exist and have length equal to number of layers
    assert hasattr(model, "caches")
    assert len(model.caches) == len(model.arch)


def test_single_backprop():
    model = make_simple_model()
    n_prev, n_curr, m = 3, 2, 4
    W_curr = np.arange(n_curr * n_prev).reshape(n_curr, n_prev).astype(float) * 0.01
    b_curr = np.zeros((n_curr, 1))
    A_prev = np.random.RandomState(2).rand(n_prev, m)
    Z_curr = W_curr @ A_prev + b_curr

    # Create small upstream dA_curr
    dA_curr = np.random.RandomState(3).rand(n_curr, m)
    dA_prev, dW_curr, db_curr = model._single_backprop(
        W_curr=W_curr,
        b_curr=b_curr,
        Z_curr=Z_curr,
        A_prev=A_prev,
        dA_curr=dA_curr,
        activation_curr="sigmoid",
    )

    # shapes
    assert dA_prev.shape == A_prev.shape
    assert dW_curr.shape == W_curr.shape
    assert db_curr.shape == b_curr.shape

    # numeric check: compute expected dZ via model._sigmoid_backprop then expected dW
    dZ_expected = model._sigmoid_backprop(dA_curr, Z_curr)
    m_val = A_prev.shape[1]
    dW_expected = (1.0 / m_val) * (dZ_expected @ A_prev.T)
    db_expected = (1.0 / m_val) * np.sum(dZ_expected, axis=1, keepdims=True)
    dAprev_expected = W_curr.T @ dZ_expected

    assert_allclose(dW_curr, dW_expected, atol=1e-8)
    assert_allclose(db_curr, db_expected, atol=1e-8)
    assert_allclose(dA_prev, dAprev_expected, atol=1e-8)

def test_predict():
    # check that forward on a sigmoid output returns values in [0,1]
    model = make_simple_model()
    X = np.random.RandomState(4).rand(3, 6)
    yhat = model.forward(X)
    yhat = np.asarray(yhat)
    assert yhat.shape == (1, X.shape[1])
    assert np.all(yhat >= 0.0) and np.all(yhat <= 1.0)


def test_binary_cross_entropy():
    model = make_simple_model()
    # simple y and y_hat
    y = np.array([[1, 0, 1]], dtype=float)
    y_hat = np.array([[0.9, 0.2, 0.1]], dtype=float)
    # compute expected BCE
    eps = 1e-8
    yh = np.clip(y_hat, eps, 1 - eps)
    expected = -np.mean(y * np.log(yh) + (1 - y) * np.log(1 - yh))
    assert_allclose(model._binary_cross_entropy(y, y_hat), expected, atol=1e-8)


def test_binary_cross_entropy_backprop():
    model = make_simple_model()
    y = np.array([[1, 0, 1]], dtype=float)
    y_hat = np.array([[0.9, 0.2, 0.1]], dtype=float)
    dA = model._binary_cross_entropy_backprop(y, y_hat)
    # manually compute bce
    eps = 1e-8
    yhat_clip = np.clip(y_hat, eps, 1 - eps)
    m = y.shape[1]
    expected = (-y / yhat_clip + (1 - y) / (1 - yhat_clip)) / m
    assert_allclose(dA, expected, atol=1e-8)


def test_mean_squared_error():
    model = make_simple_model()
    y = np.array([[1.0, 2.0]])
    y_hat = np.array([[1.2, 1.8]])
    expected = np.mean((y - y_hat) ** 2)
    assert_allclose(model._mean_squared_error(y, y_hat), expected, atol=1e-8)


def test_mean_squared_error_backprop():
    model = make_simple_model()
    y = np.array([[1.0, 2.0]])
    y_hat = np.array([[1.2, 1.8]])
    # expected derivative (2/N)*(y_hat - y)
    N = y.size
    expected = (2.0 / N) * (y_hat - y)
    assert_allclose(model._mean_squared_error_backprop(y, y_hat), expected, atol=1e-8)


def test_sample_seqs():
    seqs = ["AAA", "TTT", "CCCC", "GGG"]
    labels = [1, 0, 0, 0]
    s_seqs, s_labels = sample_seqs(seqs, labels)
    assert len(s_seqs) == len(s_labels)
    # should only return sequences containing strings
    assert all(isinstance(s, str) for s in s_seqs)


def test_one_hot_encode_seqs():
    seqs = ["AGA"]
    enc = one_hot_encode_seqs(seqs)
    enc = np.asarray(enc)
    expected_flat = np.array([1,0,0,0, 0,0,0,1, 1,0,0,0], dtype=float)
    flat = enc.reshape(-1)
    assert flat.size >= expected_flat.size
    assert_allclose(flat[:expected_flat.size], expected_flat, atol=1e-8)



def make_simple_model():
    # Helper: create a tiny NeuralNetwork for testing and return it
    nn_arch = [
        {"input_dim": 3, "output_dim": 2, "activation": "relu"},
        {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"},
    ]
    model = NeuralNetwork(
        nn_arch=nn_arch,
        lr=1e-3,
        seed=1,
        batch_size=2,
        epochs=1,
        loss_function="binary_cross_entropy",
    )
    return model