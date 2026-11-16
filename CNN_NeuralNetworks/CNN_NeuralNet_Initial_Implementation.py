'''
Code Property of: Asinak Goswami (GitHub username: asinakgoswami-hue)

Date of push: 16/11/2026 (DD/MM/YYYY)

This is a learning implementation. Feel free to add comments on what I should work on next!

'''

# ======================================== ACTUAL CODE ========================================

"""
CNN_NeuralNet.py
----------------
Intermediate pure-NumPy CNN for MNIST.

This file still contains many of the older / slower implementations
(commented out) so the evolution of the vectorized convolution and
pooling can be seen.

Architecture (final active version):
    Conv(8, 3×3) → ReLU → MaxPool(2×2)
    Conv(16, 3×3) → ReLU → MaxPool(2×2)
    Flatten → Dense(400→64) → ReLU → Dense(64→10) → Softmax
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# =========================================================
# TESTS IN TERMINAL (OUTPUTS) 
# =========================================================
""""
Terminal Log data for rewarding test epochs:
    Documented Test 1 ( Accuracy : 99.71% ) , Time :- [23 : 14 hrs] , Date :- 25 / 9 / 26:
    Terminal: 
    {
        ========================================
        CNN TRAINING STARTING
        ========================================
        X dtype: float32
        X min: 0.0
        X max: 1.0
        X shape: (10000, 28, 28, 1)
        Y shape: (10000, 10)
        Epoch 10/30 | Cost: 0.085406 | Accuracy: 97.27%
        Epoch 20/30 | Cost: 0.036669 | Accuracy: 99.14%
        Epoch 30/30 | Cost: 0.014630 | Accuracy: 99.71%

        ========================================
                CNN TRAINING COMPLETE
        ========================================
        Training accuracy: 99.71%

        FIRST 20 PREDICTIONS
        ----------------------------------------
        Image  0 | Predicted: 5 | Actual: 5 | ✓
        Image  1 | Predicted: 0 | Actual: 0 | ✓
        Image  2 | Predicted: 4 | Actual: 4 | ✓
        Image  3 | Predicted: 1 | Actual: 1 | ✓
        Image  4 | Predicted: 9 | Actual: 9 | ✓
        Image  5 | Predicted: 2 | Actual: 2 | ✓
        Image  6 | Predicted: 1 | Actual: 1 | ✓
        Image  7 | Predicted: 3 | Actual: 3 | ✓
        Image  8 | Predicted: 1 | Actual: 1 | ✓
        Image  9 | Predicted: 4 | Actual: 4 | ✓
        Image 10 | Predicted: 3 | Actual: 3 | ✓
        Image 11 | Predicted: 5 | Actual: 5 | ✓
        Image 12 | Predicted: 3 | Actual: 3 | ✓
        Image 13 | Predicted: 6 | Actual: 6 | ✓
        Image 14 | Predicted: 1 | Actual: 1 | ✓
        Image 15 | Predicted: 7 | Actual: 7 | ✓
        Image 16 | Predicted: 2 | Actual: 2 | ✓
        Image 17 | Predicted: 8 | Actual: 8 | ✓
        Image 18 | Predicted: 6 | Actual: 6 | ✓
        Image 19 | Predicted: 9 | Actual: 9 | ✓
        batch size = 16 | number of images = 10000 |learning rate = 0.01 |number of epochs = 30 

        ========================================
                    DONE
        ========================================    
        
    }
    
    Documented Test 1 ( Training Accuracy : 99.71% ) , Time :- [23 : 14 hrs] , Date :- 25 / 9 / 26:
    Terminal:
    {
        ========================================
        CNN TRAINING STARTING
        ========================================
        X dtype: float32
        X min: 0.0
        X max: 1.0
        X shape: (5000, 28, 28, 1)
        Y shape: (5000, 10)
        
        Epoch 10/30 | Cost: 0.098140 | Accuracy: 97.66%
        Epoch 20/30 | Cost: 0.040380 | Accuracy: 99.16%
        Epoch 30/30 | Cost: 0.015115 | Accuracy: 99.86%

        ========================================
                CNN TRAINING COMPLETE
        ========================================
        Training accuracy: 99.86%

        FIRST 20 PREDICTIONS
        ----------------------------------------
        Image  0 | Predicted: 5 | Actual: 5 | ✓
        Image  1 | Predicted: 0 | Actual: 0 | ✓
        Image  2 | Predicted: 4 | Actual: 4 | ✓
        Image  3 | Predicted: 1 | Actual: 1 | ✓
        Image  4 | Predicted: 9 | Actual: 9 | ✓
        Image  5 | Predicted: 2 | Actual: 2 | ✓
        Image  6 | Predicted: 1 | Actual: 1 | ✓
        Image  7 | Predicted: 3 | Actual: 3 | ✓
        Image  8 | Predicted: 1 | Actual: 1 | ✓
        Image  9 | Predicted: 4 | Actual: 4 | ✓
        Image 10 | Predicted: 3 | Actual: 3 | ✓
        Image 11 | Predicted: 5 | Actual: 5 | ✓
        Image 12 | Predicted: 3 | Actual: 3 | ✓
        Image 13 | Predicted: 6 | Actual: 6 | ✓
        Image 14 | Predicted: 1 | Actual: 1 | ✓
        Image 15 | Predicted: 7 | Actual: 7 | ✓
        Image 16 | Predicted: 2 | Actual: 2 | ✓
        Image 17 | Predicted: 8 | Actual: 8 | ✓
        Image 18 | Predicted: 6 | Actual: 6 | ✓
        Image 19 | Predicted: 9 | Actual: 9 | ✓
        batch size = 16 | number of images = 5000 |learning rate = 0.01 |number of epochs = 30 

        ========================================
                    DONE
        ========================================
    }
    
    Documented Test 1 ( Test Accuracy : 98.69% ) , Time :- [10 : 39 hrs] , Date :- 26 / 9 / 26:
    Terminal:
    {
        ========================================
        CNN TRAINING STARTING
        ========================================
        X_train dtype: float32
        X_train min: 0.0
        X_train max: 1.0
        X_train shape: (784, 60000)
        Y_train shape: (10, 60000)
        X_test shape: (784, 10000)
        Y_test shape: (10000,)
        y_test shape: (10000,)

        Training data:
        X: (60000, 28, 28, 1)
        Y: (60000, 10)

        Test data:
        X: (10000, 28, 28, 1)
        Y: (10000,)
        Epoch 1/20 | Cost: 0.399538 | Train: 94.60% | Test: 94.85%
        Epoch 2/20 | Cost: 0.147772 | Train: 96.37% | Test: 96.65%
        Epoch 3/20 | Cost: 0.107678 | Train: 97.15% | Test: 97.46%
        Epoch 4/20 | Cost: 0.089042 | Train: 97.45% | Test: 97.16%
        Epoch 5/20 | Cost: 0.077549 | Train: 97.50% | Test: 97.67%
        Epoch 6/20 | Cost: 0.069223 | Train: 97.95% | Test: 97.83%
        Epoch 7/20 | Cost: 0.063298 | Train: 98.05% | Test: 98.10%
        Epoch 8/20 | Cost: 0.058300 | Train: 98.44% | Test: 98.37%
        Epoch 9/20 | Cost: 0.053894 | Train: 98.36% | Test: 98.05%
        Epoch 10/20 | Cost: 0.050142 | Train: 98.56% | Test: 98.33%
        Epoch 11/20 | Cost: 0.047222 | Train: 98.61% | Test: 98.21%
        Epoch 12/20 | Cost: 0.043966 | Train: 98.50% | Test: 98.26%
        Epoch 13/20 | Cost: 0.041635 | Train: 98.71% | Test: 98.35%
        Epoch 14/20 | Cost: 0.039122 | Train: 99.03% | Test: 98.64%
        Epoch 15/20 | Cost: 0.037383 | Train: 98.90% | Test: 98.52%
        Epoch 16/20 | Cost: 0.035623 | Train: 98.99% | Test: 98.50%
        Epoch 17/20 | Cost: 0.034079 | Train: 98.85% | Test: 98.50%
        Epoch 18/20 | Cost: 0.032438 | Train: 99.22% | Test: 98.78%
        Epoch 19/20 | Cost: 0.030534 | Train: 99.20% | Test: 98.63%
        Epoch 20/20 | Cost: 0.029311 | Train: 99.23% | Test: 98.69%

        Total training time: 939.15 seconds
        Average time per epoch: 46.96 seconds

        ========================================
                CNN TRAINING COMPLETE
        ========================================
        Training images: 60000
        Test images: 10000
        Test accuracy: 98.69%

        FIRST 20 TEST PREDICTIONS
        ----------------------------------------
        Image  0 | Predicted: 7 | Actual: 7 | ✓
        Image  1 | Predicted: 2 | Actual: 2 | ✓
        Image  2 | Predicted: 1 | Actual: 1 | ✓
        Image  3 | Predicted: 0 | Actual: 0 | ✓
        Image  4 | Predicted: 4 | Actual: 4 | ✓
        Image  5 | Predicted: 1 | Actual: 1 | ✓
        Image  6 | Predicted: 4 | Actual: 4 | ✓
        Image  7 | Predicted: 9 | Actual: 9 | ✓
        Image  8 | Predicted: 5 | Actual: 5 | ✓
        Image  9 | Predicted: 9 | Actual: 9 | ✓
        Image 10 | Predicted: 0 | Actual: 0 | ✓
        Image 11 | Predicted: 6 | Actual: 6 | ✓
        Image 12 | Predicted: 9 | Actual: 9 | ✓
        Image 13 | Predicted: 0 | Actual: 0 | ✓
        Image 14 | Predicted: 1 | Actual: 1 | ✓
        Image 15 | Predicted: 5 | Actual: 5 | ✓
        Image 16 | Predicted: 9 | Actual: 9 | ✓
        Image 17 | Predicted: 7 | Actual: 7 | ✓
        Image 18 | Predicted: 8 | Actual: 3 | ✗
        Image 19 | Predicted: 4 | Actual: 4 | ✓
        batch size = 32 | training images = 60000 | learning rate = 0.01 | epochs = 20
    }
    
    
"""


#ReLU activation function
def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A , cache

def relu_backward(dA , Z):
    
    dZ = dA.copy()
    dZ[ Z <= 0 ] = 0
    
    return dZ

#Softmax activation function
def softmaxCNN(Z):
    expZ = np.exp(
        Z - np.max(Z , axis = 1 , keepdims = True)
    )
    cache = Z
    return expZ / np.sum(
        expZ , axis = 1 , keepdims = True
    ) , cache

# New cost function (NOT binary cross entropy , instead categorical cross entropy)
# As the MNIST dataset has 10 classes

# ============================================================
# CNN COST FUNCTION
# ============================================================

def cost_function(A, Y):

    m = Y.shape[0]

    epsilon = 1e-12

    cost = -(1 / m) * np.sum(
        Y * np.log(A + epsilon)
    )

    return cost


''''
Old , broken constructor using a standard weight for all layers

def init_params():
    
    params = {
        # ---------------- CONV 1 ----------------

        "Wc1": np.random.randn(
            8, 3, 3, 1
        ) * 0.01,

        "bc1": np.zeros(8),


        # ---------------- CONV 2 ----------------

        "Wc2": np.random.randn(
            16, 3, 3, 8
        ) * 0.01,

        "bc2": np.zeros(16),


        # ---------------- DENSE 1 ----------------

        "W3": np.random.randn(
            400, 64
        ) * 0.01,

        "b3": np.zeros(64),


        # ---------------- DENSE 2 ----------------

        "W4": np.random.randn(
            64, 10
        ) * 0.01,

        "b4": np.zeros(10)
    }

    return params
'''    
    
# =========================================================================
# New Constructor with different values as weights using HE initialization  
# =========================================================================

'''
# This constructor is memory inefficient as defaults are float64

def init_params():
    params = {

        # Conv1: fan_in = 3*3*1 = 9
        "Wc1": np.random.randn(8, 3, 3, 1) * np.sqrt(2 / 9),
        "bc1": np.zeros(8),

        # Conv2: fan_in = 3*3*8 = 72
        "Wc2": np.random.randn(16, 3, 3, 8) * np.sqrt(2 / 72),
        "bc2": np.zeros(16),

        # Dense: fan_in = 5*5*16 = 400
        "W3": np.random.randn(400, 64) * np.sqrt(2 / 400),
        "b3": np.zeros(64),

        # Dense: fan_in = 64
        "W4": np.random.randn(64, 10) * np.sqrt(2 / 64),
        "b4": np.zeros(10)
    }
    
    return params
'''

# This constructor will make sure that all weights and biases start as float32

def init_params():

    params = {

        "Wc1": (
            np.random.randn(8,3,3,1) *
            np.sqrt(2/9)
        ).astype(np.float32),

        "bc1": np.zeros(8, dtype=np.float32),


        "Wc2": (
            np.random.randn(16,3,3,8) *
            np.sqrt(2/72)
        ).astype(np.float32),

        "bc2": np.zeros(16, dtype=np.float32),


        "W3": (
            np.random.randn(400,64) *
            np.sqrt(2/400)
        ).astype(np.float32),

        "b3": np.zeros(64, dtype=np.float32),


        "W4": (
            np.random.randn(64,10) *
            np.sqrt(2/64)
        ).astype(np.float32),

        "b4": np.zeros(10, dtype=np.float32)
    }

    return params

# ============================================================
# FORWARD PROPAGATION
# ============================================================

def forward_prop(X , params):
    # -------------- CONV 1 --------------
    
    Z1 , conv1_cache = conv2D_forward(
        X ,
        params["Wc1"],
        params["bc1"]
    )
    
    #-------------- ReLU 1 --------------
    
    A1 , relu1_cache = relu(Z1)
    
    #-------------- POOL 1 --------------
    
    P1 , pool1_cache = max_pool_forward(
        A1,
        pool_size=2,
        stride=2
    )
    
    # -------------- CONV 2 --------------
    
    Z2 , conv2_cache = conv2D_forward(
        P1 , 
        params["Wc2"],
        params["bc2"]
    )
    
    #-------------- ReLu 2 --------------
    
    A2 , relu2_cache = relu(Z2)
    
    #-------------- POOL 2 --------------
    
    P2 , pool2_cache = max_pool_forward(
        A2 , 
        pool_size=2,
        stride=2
    )
    
    # -------------- FLATTEN LAYERS --------------
    F, flatten_cache = flatten(P2)
    
    
    Z3 = F @ params["W3"] + params["b3"]
    
    A3 , relu3_cache = relu(Z3)
    
    Z4 = A3 @ params["W4"] + params["b4"]
    
    Y_hat , softmaxx_cache = softmaxCNN(Z4)
    
    caches = {
        
            "Z1": Z1,
            "A1": A1,
            "P1": P1,

            "Z2": Z2,
            "A2": A2,
            "P2": P2,

            "F": F,

            "Z3": Z3,
            "A3": A3,

            "Z4": Z4,

            "conv1": conv1_cache,
            "relu1": relu1_cache,
            "pool1": pool1_cache,

            "conv2": conv2_cache,
            "relu2": relu2_cache,
            "pool2": pool2_cache,

            "flatten": flatten_cache,

            "relu3": relu3_cache,
            "softmax": softmaxx_cache
    }

    return Y_hat, caches

# helper dense method:
def dense_backward(dZ , A , W):
    dW = A.T @ dZ
    
    db = np.sum(
        dZ ,
        axis =  0
    )

    dA = dZ @ W.T

    return dA , dW , db


def backprop(Y_hat , Y , caches , params):
    
    m = Y.shape[0]
    
    dZ4 = (Y_hat - Y) / m

    dA3 , dW4 , db4 = dense_backward(
        dZ4 , 
        caches["A3"],
        params["W4"]
    )
    
    dZ3 = relu_backward(
        dA3 , 
        caches["Z3"]
    )
    
    dF , dW3 , db3 = dense_backward(
        dZ3 ,
        caches["F"],
        params["W3"]
    )
    
    dP2 = unflatten(
        dF ,
        caches["flatten"]
    )
    
    dA2 = max_pool_backward(
        dP2 ,
        caches["pool2"]
    )
    
    dZ2 = relu_backward(
        dA2 ,
        caches["Z2"]
    )
    
    dP1 , dWc2, dbc2 = conv2D_backward(
        dZ2,
        caches["conv2"]
    )
    
    dA1 = max_pool_backward(
        dP1 ,
        caches["pool1"]
    )
    
    dZ1 = relu_backward(
        dA1 , 
        caches["Z1"]
    )
    
    dX , dWc1 , dbc1 = conv2D_backward(
        dZ1 ,
        caches["conv1"]
    )
    
    # storage of the gradients
    
    grads = {

        "dWc1": dWc1,
        "dbc1": dbc1,

        "dWc2": dWc2,
        "dbc2": dbc2,

        "dW3": dW3,
        "db3": db3,

        "dW4": dW4,
        "db4": db4,
        
        "dX" : dX 
    }

    return grads

# ============================================================
# CNN PARAMETER UPDATE
# ============================================================

def update_params(params, grads, learning_rate):

    # ---------------- CONV 1 ----------------

    params["Wc1"] -= (
        learning_rate * grads["dWc1"]
    )

    params["bc1"] -= (
        learning_rate * grads["dbc1"]
    )


    # ---------------- CONV 2 ----------------

    params["Wc2"] -= (
        learning_rate * grads["dWc2"]
    )

    params["bc2"] -= (
        learning_rate * grads["dbc2"]
    )


    # ---------------- DENSE 1 ----------------

    params["W3"] -= (
        learning_rate * grads["dW3"]
    )

    params["b3"] -= (
        learning_rate * grads["db3"]
    )


    # ---------------- DENSE 2 ----------------

    params["W4"] -= (
        learning_rate * grads["dW4"]
    )

    params["b4"] -= (
        learning_rate * grads["db4"]
    )


    return params

# ============================================================
# CNN eval method using only 1 forward prop mechanism
# ============================================================

def evaluate(X, Y, params, batch_size=128):

    m = X.shape[0]

    correct = 0
    total_cost = 0.0

    for start in range(0, m, batch_size):

        end = min(start + batch_size, m)

        X_batch = X[start:end]
        Y_batch = Y[start:end]

        Y_hat, _ = forward_prop(
            X_batch,
            params
        )

        predictions = np.argmax(
            Y_hat,
            axis=1
        )

        actual = np.argmax(
            Y_batch,
            axis=1
        )

        correct += np.sum(
            predictions == actual
        )

        batch_cost = cost_function(
            Y_hat,
            Y_batch
        )

        total_cost += batch_cost * (end - start)

    accuracy = (correct / m) * 100
    cost = total_cost / m

    return cost, accuracy

# ============================================================
# CNN TRAINING LOOP
# ============================================================

'''
# Old train method
def train(
    X,
    Y,
    epochs,
    learning_rate,
    batch_size,
    print_every=1
):

    # --------------------------------------------------------
    # Initialize CNN parameters
    # --------------------------------------------------------

    params = init_params()

    cost_history = []

    m = X.shape[0]


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(epochs):

        # ----------------------------------------------------
        # Shuffle dataset
        # ----------------------------------------------------

        permutation = np.random.permutation(m)

        X_shuffled = X[permutation]
        Y_shuffled = Y[permutation]


        epoch_cost = 0.0


        # ====================================================
        # MINI-BATCH LOOP
        # ====================================================

        for start in range(
            0,
            m,
            batch_size
        ):

            end = min(
                start + batch_size,
                m
            )


            # ------------------------------------------------
            # Create batch
            # ------------------------------------------------

            X_batch = X_shuffled[
                start:end
            ]

            Y_batch = Y_shuffled[
                start:end
            ]


            # ------------------------------------------------
            # FORWARD PROPAGATION
            # ------------------------------------------------

            Y_hat, caches = forward_prop(
                X_batch,
                params
            )


            # ------------------------------------------------
            # COST
            # ------------------------------------------------

            cost = cost_function(
                Y_hat,
                Y_batch
            )


            # Weighted cost contribution
            batch_m = X_batch.shape[0]

            epoch_cost += (
                cost * batch_m
            )


            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            grads = backprop(
                Y_hat,
                Y_batch,
                caches,
                params
            )


            # ------------------------------------------------
            # UPDATE PARAMETERS
            # ------------------------------------------------

            params = update_params(
                params,
                grads,
                learning_rate
            )


        # ====================================================
        # END OF EPOCH
        # ====================================================

        epoch_cost /= m

        cost_history.append(
            epoch_cost
        )


        # ----------------------------------------------------
        # Training accuracy
        # ----------------------------------------------------

        if (epoch+1) % print_every == 0:

            predictions, probabilities = predict(
                X,
                params
            )

            actual = np.argmax(
                Y,
                axis=1
            )

            accuracy = np.mean(
                predictions == actual
            ) * 100


            print(
                f"Epoch {epoch + 1}/{epochs}"
                f" | Cost: {epoch_cost:.6f}"
                f" | Accuracy: {accuracy:.2f}%"
            )
            
        else:
            print(
                f"Epoch {epoch + 1} / {epochs}"
                f" | Cost: {epoch_cost:.6f}"
            )
            


    return params, cost_history


# ============================================================
# CNN PREDICTION
# ============================================================
'''

# New Train() method
def train(
    X,
    Y,
    epochs,
    learning_rate,
    batch_size,
    print_every=1,
    X_test=None,
    Y_test=None,
    eval_batch_size=128
):

    params = init_params()

    cost_history = []

    m = X.shape[0]

    for epoch in range(epochs):

        # =========================================================
        # Shuffle training data
        # =========================================================

        permutation = np.random.permutation(m)

        X_shuffled = X[permutation]
        Y_shuffled = Y[permutation]

        epoch_cost = 0.0


        # =========================================================
        # Mini-batch training
        # =========================================================

        for start in range(0, m, batch_size):

            end = min(
                start + batch_size,
                m
            )

            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            # -----------------------------------------------------
            # Forward propagation
            # -----------------------------------------------------

            Y_hat, caches = forward_prop(
                X_batch,
                params
            )

            # -----------------------------------------------------
            # Cost
            # -----------------------------------------------------

            cost = cost_function(
                Y_hat,
                Y_batch
            )

            current_batch_size = X_batch.shape[0]

            epoch_cost += (
                cost * current_batch_size
            )

            # -----------------------------------------------------
            # Backpropagation
            # -----------------------------------------------------

            grads = backprop(
                Y_hat,
                Y_batch,
                caches,
                params
            )

            # -----------------------------------------------------
            # Update parameters
            # -----------------------------------------------------

            params = update_params(
                params,
                grads,
                learning_rate
            )


        # =========================================================
        # Average cost for this epoch
        # =========================================================

        epoch_cost /= m

        cost_history.append(
            epoch_cost
        )


        # =========================================================
        # Evaluation
        # =========================================================

        if (epoch + 1) % print_every == 0:

            # -----------------------------------------------------
            # Training accuracy
            # -----------------------------------------------------

            train_predictions, _ = predict(
                X,
                params,
                batch_size=eval_batch_size
            )

            train_actual = np.argmax(
                Y,
                axis=1
            )

            train_accuracy = (
                np.mean(
                    train_predictions == train_actual
                ) * 100
            )


            # -----------------------------------------------------
            # Test accuracy
            # -----------------------------------------------------

            if X_test is not None and Y_test is not None:

                test_predictions, _ = predict(
                    X_test,
                    params,
                    batch_size=eval_batch_size
                )

                test_actual = Y_test

                test_accuracy = (
                    np.mean(
                        test_predictions == test_actual
                    ) * 100
                )

                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Cost: {epoch_cost:.6f} | "
                    f"Train: {train_accuracy:.2f}% | "
                    f"Test: {test_accuracy:.2f}%"
                )


    return params, cost_history


'''
# Old predict()
def predict(X, params):

    # Run the image(s) through the network
    AL, caches = forward_prop(X, params)

    # Find the output neuron with the highest probability
    predictions = np.argmax(AL, axis=1)

    return predictions, AL

'''

# New predict()
def predict(X, params, batch_size=128):

    all_predictions = []
    all_probabilities = []

    for start in range(0, X.shape[0], batch_size):

        end = min(start + batch_size, X.shape[0])

        X_batch = X[start:end]

        Y_hat, caches = forward_prop(
            X_batch,
            params
        )

        predictions = np.argmax(Y_hat, axis=1)

        all_predictions.append(predictions)
        all_probabilities.append(Y_hat)

    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)

    return predictions, probabilities

# ==================================================
# CNN methods (useful for forward prop and backprop) 
# ==================================================

'''

# Old convolution methods, REALLY SLOW !
def conv2D_forward(X , W , b , stride = 1 , padding = 0):
    
    #m      → number of images
    #H      → image height
    #W_in   → image width
    #C      → input channels
    
    m , H , W_in , C = X.shape
    
    # Assuming W.shape = (num_filters, filter_height, filter_width, channels)
    F = W.shape[1]
    num_filters = W.shape[0]
    
    # Output dimensions 
    H_out = (H + 2 * padding - F ) // stride + 1 
    W_out = (W_in + 2 * padding - F ) // stride + 1 
    
    # Output tensor
    Z = np.zeros(
        (m , H_out , W_out , num_filters)
    )
    # Proper padding of the image array using np.pad()
    X_pad = np.pad(
        X , 
        (
            (0,0),
            (padding , padding) , 
            (padding , padding) , 
            (0,0)
        ) , 
        mode = "constant"
    )
    
    for n in range(m): # Process each image in the batch
        
        for f in range(num_filters): # Process each filter
            
            for i in range(H_out):  # Move vertically
                
                for j in range(W_out): # Move horizontally
                    
                    h_start = i*stride
                    h_end = h_start + F
                    
                    w_start = j * stride
                    w_end = w_start + F
                    
                    patch = X_pad[
                        n , 
                        h_start:h_end ,
                        w_start:w_end ,
                        :
                    ]
                    
                    Z[n , i , j , f] = (
                        np.sum(
                            patch * W[f]
                        ) 
                        + b[f]
                    )
                    
                    
    cache = (
        X ,
        W ,
        b ,
        stride ,
        padding
    )
    
    return Z , cache
    
 
def conv2D_backward(dZ , cache):
    #The cache from the Conv2d_forward contains X , W , b vectors 
    
    X , W , b , stride , padding = cache
    
    # m      → number of images
    # H      → image height
    # W_in   → image width
    # C      → input channels
    
    m , H , W_in , C = X.shape
    
    num_filters , F , _ , _ = W.shape
    
    H_out = dZ.shape[1]
    W_out = dZ.shape[2]
    
    #Pad X
    X_pad = np.pad(
        X , 
        (
            (0,0),
            (padding , padding) , 
            (padding , padding) , 
            (0,0)
            
        ) , mode = "constant"
    )
    
    
    # initializing the gradients dX , dW , db
    dX_pad = np.zeros_like(X_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    
   
    for n in range(m): # Calculate gradients in each image
            
            for f in range(num_filters): # Calculate gradients in each image
                
                for i in range(H_out):  # Move vertically
                    
                    for j in range(W_out): # Move horizontally
                        
                        h_start = i*stride
                        h_end = h_start + F
                        
                        w_start = j * stride
                        w_end = w_start + F
                        
                        patch = X_pad[
                            n , 
                            h_start:h_end ,
                            w_start:w_end ,
                            :
                        ]
                        
                        #getting the incoming gradient
                        dz = dZ[n,i,j,f]
                        
                        #Calculate gradient of weights
                        dW += patch * dz
                        
                        #Calculate gradient of biases
                        db += dz
                        
                        #Calculate dX
                        dX_pad[
                            n , 
                            h_start : h_end , 
                            w_start : w_end,
                            :
                        ] += W[f] * dz
    
    if padding == 0 : 
        dX = dX_pad
    else:
        dX = dX_pad[
            : , 
            padding:-padding,
            padding:-padding,
            :
        ] 
    return dX , dW , db
    
    
'''

#New Convolution methods , SHOULD BE FASTER
def conv2D_forward(X , W , b , stride = 1 , padding = 0):
    
    #m      → number of images
    #H      → image height
    #W_in   → image width
    #C      → input channels
    
    m , H , W_in , C = X.shape
    
    # Assuming W.shape = (num_filters, filter_height, filter_width, channels)
    F = W.shape[1]
    num_filters = W.shape[0]
    
    # Output dimensions 
    H_out = (H + 2 * padding - F ) // stride + 1 
    W_out = (W_in + 2 * padding - F ) // stride + 1 
    
    X_pad = np.pad(
        X , 
        (
            (0,0) , 
            (padding , padding),
            (padding , padding),
            (0,0)
        ) , mode = "constant"
    )
    
    patches = np.lib.stride_tricks.sliding_window_view(
        X_pad,
        (F , F) ,
        axis = (1 , 2)
    )
    
    # Shape produced:
    # (m, H_out, W_out, C, F, F)

    patches = patches[:, ::stride, ::stride, :, :, :]
    
    # Rearrange : (m , H_out , W_out , F ,F ,C)
    
    patches = patches.transpose(
        0,1,2,4,5,3
    )
    
    #Flatten patches
    
    patches_flat = patches.reshape(
        m * H_out * W_out,
        F * F * C
    )
    
    # Flatten filters
    
    W_flat = W.reshape(
        num_filters , 
        F * F * C
    )
    
    # matrix multiplication
    
    Z = patches_flat @ W_flat.T

    # add bias
    
    Z += b    
    
    
    # Restore Image structure
    
    Z = Z.reshape(
        m ,
        H_out,
        W_out,
        num_filters
    )
    
    cache = (
        X ,
        W,
        b,
        stride,
        padding,
        patches
    )

    return Z , cache

'''
# older convolution backward function 

def conv2D_backward(dZ, cache):

    X, W, b, stride, padding, patches = cache

    m, H, W_in, C = X.shape

    num_filters, F, _, _ = W.shape

    H_out = dZ.shape[1]
    W_out = dZ.shape[2]

    # --------------------------------------------------------
    # Flatten everything
    # --------------------------------------------------------

    dZ_flat = dZ.reshape(
        m * H_out * W_out,
        num_filters
    )

    patches_flat = patches.reshape(
        m * H_out * W_out,
        F * F * C
    )

    W_flat = W.reshape(
        num_filters,
        F * F * C
    )

    # --------------------------------------------------------
    # Gradient of filters
    # --------------------------------------------------------

    dW_flat = dZ_flat.T @ patches_flat

    dW = dW_flat.reshape(
        W.shape
    )

    # --------------------------------------------------------
    # Gradient of biases
    # --------------------------------------------------------

    db = np.sum(
        dZ,
        axis=(0, 1, 2)
    )

    # --------------------------------------------------------
    # Gradient with respect to input patches
    # --------------------------------------------------------

    dpatches_flat = dZ_flat @ W_flat

    dpatches = dpatches_flat.reshape(
        m,
        H_out,
        W_out,
        F,
        F,
        C
    )

    # --------------------------------------------------------
    # Reconstruct dX
    # --------------------------------------------------------

    H_pad = H + 2 * padding
    W_pad = W_in + 2 * padding

    dX_pad = np.zeros(
        (m, H_pad, W_pad, C),
        dtype=np.float32
    )

    for i in range(H_out):

        for j in range(W_out):

            h_start = i * stride
            h_end = h_start + F

            w_start = j * stride
            w_end = w_start + F

            dX_pad[
                :,
                h_start:h_end,
                w_start:w_end,
                :
            ] += dpatches[
                :,
                i,
                j,
                :,
                :,
                :
            ]

    # --------------------------------------------------------
    # Remove padding
    # --------------------------------------------------------

    if padding == 0:

        dX = dX_pad

    else:

        dX = dX_pad[
            :,
            padding:-padding,
            padding:-padding,
            :
        ]

    return dX, dW, db

'''

def conv2D_backward(dZ, cache):

    X, W, b, stride, padding, patches = cache

    m, H, W_in, C = X.shape
    num_filters, F, _, _ = W.shape

    H_out = dZ.shape[1]
    W_out = dZ.shape[2]

    # -------------------------
    # 1. Reshape gradients
    # -------------------------

    dZ_flat = dZ.reshape(
        m * H_out * W_out,
        num_filters
    )

    patches_flat = patches.reshape(
        m * H_out * W_out,
        F * F * C
    )

    W_flat = W.reshape(
        num_filters,
        F * F * C
    )

    # -------------------------
    # 2. Gradient of weights
    # -------------------------

    dW_flat = dZ_flat.T @ patches_flat

    dW = dW_flat.reshape(W.shape)

    # -------------------------
    # 3. Gradient of biases
    # -------------------------

    db = np.sum(dZ, axis=(0, 1, 2))

    # -------------------------
    # 4. Gradient with respect
    #    to convolution patches
    # -------------------------

    dpatches_flat = dZ_flat @ W_flat

    dpatches = dpatches_flat.reshape(
        m,
        H_out,
        W_out,
        F,
        F,
        C
    )

    # -------------------------
    # 5. Reconstruct dX
    # -------------------------

    H_pad = H + 2 * padding
    W_pad = W_in + 2 * padding

    dX_pad = np.zeros(
        (m, H_pad, W_pad, C),
        dtype=X.dtype
    )

    # Vectorized spatial accumulation
    for fi in range(F):
        for fj in range(F):

            dX_pad[
                :,
                fi:fi + H_out * stride:stride,
                fj:fj + W_out * stride:stride,
                :
            ] += dpatches[:, :, :, fi, fj, :]

    # -------------------------
    # 6. Remove padding
    # -------------------------

    if padding == 0:
        dX = dX_pad
    else:
        dX = dX_pad[
            :,
            padding:-padding,
            padding:-padding,
            :
        ]

    return dX, dW, db

'''
# older max_pool_forward function() unoptimized

def max_pool_forward(X , pool_size = 2 , stride = 2):
    
    # m      → number of images
    # H      → image height
    # W_in   → image width
    # C      → input channels
    m, H, W, C = X.shape
    
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    windows = np.lib.stride_tricks


    A = np.zeros(
        (m , H_out , W_out , C)
    )
    
    for n in range(m): # Calculate gradients in each image
                
        for c in range(C): # Calculate gradients in each image
            
            for i in range(H_out):  # Move vertically
                
                for j in range(W_out): # Move horizontally
                    
                    h_start = i * stride
                    h_end = h_start + pool_size
                    
                    w_start = j * stride
                    w_end = w_start + pool_size
                    
                    patch = X[
                        n , 
                        h_start:h_end ,
                        w_start:w_end ,
                        c
                    ]
                    
                    A[n , i , j , c] = np.max(patch)
                    
    cache = (
        X , 
        pool_size , 
        stride
    )
    
    return A , cache
'''

def max_pool_forward(A, pool_size=2, stride=2):

    m, H, W, C = A.shape

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    # Extract all pooling windows
    windows = np.lib.stride_tricks.sliding_window_view(
        A,
        (pool_size, pool_size),
        axis=(1, 2)
    )

    windows = windows[
        :,
        ::stride,
        ::stride,
        :,
        :,
        :
    ]

    # Maximum over pooling dimensions
    P = np.max(
        windows,
        axis=(4, 5)
    )

    cache = (
        A,
        windows,
        P,
        pool_size,
        stride
    )

    return P, cache

'''
# Older max_pool_backward() function unoptimized

def max_pool_backward(dA , cache):
    X , pool_size , stride = cache
    
    m , H , W , C = X.shape
    
    H_out = dA.shape[1]
    W_out = dA.shape[2]
    
    dX = np.zeros_like(X)
    
    for n in range(m):

        for c in range(C):

            for i in range(H_out):

                for j in range(W_out):

                    h_start = i * stride
                    h_end = h_start + pool_size

                    w_start = j * stride
                    w_end = w_start + pool_size

                    patch = X[
                        n,
                        h_start:h_end,
                        w_start:w_end,
                        c
                    ]
                    
                    #Find the maximum
                    max_value = np.max(patch)
                    
                    #create mask
                    mask = (patch == max_value)

                    #Gradient flowing from the pooling output
                    da = dA[n , i , j ,c]
                    
                    #Send gradient only to maximum
                    dX[
                        n , 
                        h_start:h_end,
                        w_start:w_end,
                        c
                    ] += mask * da
    
    return dX 

'''

# New max_pool_backward function() optimized
def max_pool_backward(dP, cache):

    A, windows, P, pool_size, stride = cache

    m, H, W, C = A.shape
    H_out = dP.shape[1]
    W_out = dP.shape[2]

    dA = np.zeros_like(A)

    # Flatten each pooling window
    windows_flat = windows.reshape(
        m,
        H_out,
        W_out,
        C,
        pool_size * pool_size
    )

    # Position of maximum
    max_idx = np.argmax(
        windows_flat,
        axis=4
    )

    # Convert flattened index to row/column
    max_i = max_idx // pool_size
    max_j = max_idx % pool_size

    # --------------------------------
    # Coordinates of pooling windows
    # --------------------------------

    batch_idx = np.arange(m)[:, None, None, None]

    row_base = (
        np.arange(H_out)[None, :, None, None]
        * stride
    )

    col_base = (
        np.arange(W_out)[None, None, :, None]
        * stride
    )

    channel_idx = np.arange(C)[None, None, None, :]

    # --------------------------------
    # Scatter gradients
    # --------------------------------

    np.add.at(
        dA,
        (
            batch_idx,
            row_base + max_i,
            col_base + max_j,
            channel_idx
        ),
        dP
    )

    return dA

def flatten(X):
    
    m = X.shape[0]
    A = X.reshape(m, -1)
    cache = X.shape
    
    return A , cache

def unflatten(dA , cache):
    
    original_shape = cache
    dX = dA.reshape(original_shape)
    
    return dX


'''
# ================== Testing and Execution Area ==================

# ============================================================
# CNN QUICK EXECUTION
# ============================================================

print("\n========================================")
print("        CNN TRAINING STARTING")
print("========================================")

X_train = np.load("X_train.npy").astype(np.float32)
Y_train = np.load("Y_train.npy")

X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")
y_test = np.load("y_test.npy")

# ------------------------------------------------------------
# Prepare 1000 MNIST images for CNN
# ------------------------------------------------------------

print("X dtype:", X_train.dtype)
print("X min:", X_train.min())
print("X max:", X_train.max())

num_of_images = 1000

X_test_run = X_train[:,:num_of_images].T.reshape(
    -1,
    28,
    28,
    1
)

# ------------------------------------------------------------
# Convert labels:
#
# Old format:
#       (10, samples)
#
# CNN format:
#       (samples, 10)
# ------------------------------------------------------------

Y_test_run = Y_train[:, :num_of_images].T

print("X shape:", X_test_run.shape)
print("Y shape:", Y_test_run.shape)

# ------------------------------------------------------------
# Train and test
# ------------------------------------------------------------

params = init_params()

Y_hat, caches = forward_prop(X_test_run, params)

batchSize = 16
learningRate = 0.01
numEpochs = 30

params, cost_history = train(
    X_test_run,
    Y_test_run,
    epochs=numEpochs,
    learning_rate=learningRate,
    batch_size=batchSize,
    print_every=1
)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

predictions, probabilities = predict(
    X_test_run,
    params
)

actual = np.argmax(
    Y_test_run,
    axis=1
)

accuracy = np.mean(
    predictions == actual
) * 100

print("\n========================================")
print("        CNN TRAINING COMPLETE")
print("========================================")

print(
    f"Training accuracy: {accuracy:.2f}%"
)

# ------------------------------------------------------------
# Show first 20 predictions
# ------------------------------------------------------------

print("\nFIRST 20 PREDICTIONS")
print("----------------------------------------")

for i in range(20):

    print(
        f"Image {i:2d} | "
        f"Predicted: {predictions[i]} | "
        f"Actual: {actual[i]} | "
        f"{'✓' if predictions[i] == actual[i] else '✗'}"
    )

print(
    f"batch size = {batchSize} | "
    f"number of images = {num_of_images} |"
    f"learning rate = {learningRate} |"
    f"number of epochs = {numEpochs} "
)

print("\n========================================")
print("             DONE")
print("========================================")

plt.plot(cost_history)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.yscale("log")  # logarithmic scale graph
plt.title("Training Cost (see VScode terminal for data during training)")
plt.get_current_fig_manager().set_window_title(
    "Cost(y) vs Epochs(x) graph for the simple data set (see vscode terminal)"
)
plt.show()

'''



# ================== Testing and Execution Area ==================

print("\n========================================")
print("        CNN TRAINING STARTING")
print("========================================")


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

X_train = np.load("X_train.npy").astype(np.float32)
Y_train = np.load("Y_train.npy").astype(np.float32)

X_test = np.load("X_test.npy").astype(np.float32)
Y_test = np.load("Y_test.npy")
y_test = np.load("y_test.npy")


print("X_train dtype:", X_train.dtype)
print("X_train min:", X_train.min())
print("X_train max:", X_train.max())

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)
print("y_test shape:", y_test.shape)

# ------------------------------------------------------------
# Prepare training data
# ------------------------------------------------------------

num_of_images = 60000

X_train_run = X_train[:, :num_of_images].T.reshape(
    -1,
    28,
    28,
    1
)

Y_train_run = Y_train[:, :num_of_images].T

print("\nTraining data:")
print("X:", X_train_run.shape)
print("Y:", Y_train_run.shape)

# ------------------------------------------------------------
# Prepare REAL test data
# ------------------------------------------------------------

X_test_run = X_test.T.reshape(
    -1,
    28,
    28,
    1
)

Y_test_run = Y_test.T

print("\nTest data:")
print("X:", X_test_run.shape)
print("Y:", Y_test_run.shape)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

batchSize = 32
learningRate = 0.01
numEpochs = 20

start = time.time()

params, cost_history = train(
    X_train_run,
    Y_train_run,

    epochs=numEpochs,
    learning_rate=learningRate,
    batch_size=batchSize,

    print_every=1,

    X_test=X_test_run,
    Y_test=Y_test_run,

    eval_batch_size=128
)

end = time.time()

# For the time taken in total training
print(f"\nTotal training time: {end - start:.2f} seconds")
print(f"Average time per epoch: {(end - start) / numEpochs:.2f} seconds")

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

predictions, probabilities = predict(
    X_test_run,
    params
)

actual = y_test

test_accuracy = np.mean(
    predictions == actual
) * 100

print("\n========================================")
print("        CNN TRAINING COMPLETE")
print("========================================")

print(
    f"Training images: {num_of_images}"
)

print(
    f"Test images: {X_test_run.shape[0]}"
)

print(
    f"Test accuracy: {test_accuracy:.2f}%"
)

# ------------------------------------------------------------
# First 20 TEST predictions
# ------------------------------------------------------------

print("\nFIRST 20 TEST PREDICTIONS")
print("----------------------------------------")

for i in range(20):

    print(
        f"Image {i:2d} | "
        f"Predicted: {predictions[i]} | "
        f"Actual: {actual[i]} | "
        f"{'✓' if predictions[i] == actual[i] else '✗'}"
    )


print(
    f"batch size = {batchSize} | "
    f"training images = {num_of_images} | "
    f"learning rate = {learningRate} | "
    f"epochs = {numEpochs}"
)


print("\n========================================")
print("             DONE")
print("========================================")


plt.plot(cost_history)

plt.xlabel("Epoch")
plt.ylabel("Cost")

plt.yscale("log")

plt.title(
    "Training Cost vs Epochs"
)

plt.get_current_fig_manager().set_window_title( # type: ignore
    "CNN Training Cost vs Epochs"
)

plt.show()
