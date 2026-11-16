'''
Code Property of: Asinak Goswami (GitHub username: asinakgoswami-hue)

Date of push: 16/11/2026 (DD/MM/YYYY)

This is a learning implementation. Feel free to add comments on what I should work on next!

'''

# ======================================== ACTUAL CODE ======================================== 

"""
MNIST_Softmax_ReLU_NeuralNet.py
-------------------------------
Fully-connected network for MNIST using ReLU in the hidden layers
and Softmax + categorical cross-entropy at the output.

This version consistently reaches ~98 % test accuracy with the simple
architecture 784 → 64 → 10.

All original experimental logs and older training loops are kept.
"""

'''
* Tasks: 
    - To improve the accuracy of the network to better standards
    - To implement more neurons such as a 64 x 64 grid from a picture of my own handwriting
    - To make a [4096 , 128 , 64 , 32 , 10] network to read and guess my own handwriting.
    - Build standalone image → prediction pipeline
    

LOGS ::::-
Implemented Ouput-Softmax Hidden-ReLU [OSHR] activation pattern Neural Network 
------------------------------------------------------------------------------------------------
Experiment | Architecture |HL Activation | Data | Batch | Epochs | LR | Accuracy | Date
------------------------------------------------------------------------------------------------
01         | 784→64→10     | Sigmoid    | 10k  | 16    | 20       | .1 | 93.70% | 24 / 9 / 26
02         | 784→128→64→10 | Sigmoid    | 10k  | 16    | 30       | .1 | 96.59% | 24 / 9 / 26
03         | 784→64→10     | ReLU       | 15k  | 16    | 50       | .1 | 96.63% | 24 / 9 / 26
04 (*)     | 784→64→10     | ReLU       | 60k  | 16    | 50       | .1 | 98.06% | 24 / 9 / 26
05         | 784→64→10     | Sigmoid    | 60k  | 16    | 50       | .1 | 97.68% | 24 / 9 / 26
06 (*)     | 784→64→10     | ReLU       | 60k  | 16    | 50       | .1 | 97.96% | 24 / 9 / 26
07         | 784→64→10     | ReLU       | 60k  | 16    | 50       | .1 | 97.81% | 24 / 9 / 26
08 (*)     | 784→64→10     | ReLU       | 60k  | 16    | 50       | .2 | 97.93% | 24 / 9 / 26



POINTERS!! :

- This model has been tested by many different learning rates out of which Learning_rate = 0.1 
    prevails as the most usable rate for constant learning of the network 

- The epoch number of 50 is considered as the baseline operations giving maximum output for 
    minimum input and all that stuff (I forgor the rest ......)



Terminal Log data for rewarding test epochs:
    Documented Test 1 ( Accuracy : 97.96% (~) ) , Time :- [18 : 32 hrs] , Date :- 24 / 9 / 26:
    Terminal: 
    {
        Epoch: 0 | Cost: 0.2410607692878175
        Epoch: 10 | Cost: 0.0255238997127441
        Epoch: 20 | Cost: 0.002729646855038636
        Epoch: 30 | Cost: 0.0005330051883455579
        Epoch: 40 | Cost: 0.0003594851323612785
        The guess values are: 
        [7 2 1 0 4 1 4 9 5 9]
        The real values are: 
        [7 2 1 0 4 1 4 9 5 9]
        Accuracy:  97.96000000000001  %
        The size of each batch:  16
        The total number of epochs:  50
        Neuron layer dimensions : [784 -> 64 -> 10]
        Dataset size: 60000
        learning rate = 0.1
    }

    Documented Test 2 ( Accuracy : 97.93% (~) ) , Time :- [18 : 45 hrs] , Date :- 24 / 9 / 26:
    Terminal:
    {
        Epoch: 0 | Cost: 0.2414736417723024
        Epoch: 10 | Cost: 0.03192835031622677
        Epoch: 20 | Cost: 0.010687987257700248
        Epoch: 30 | Cost: 0.00024362655787153924
        Epoch: 40 | Cost: 0.00015069377435192004
        The guess values are: 
        [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4 9 6 6 5 4 0 7 4 0 1 3 1 3 4 7 2 7
        1 2 1 1 7 4 2 3 5 1 2 4 4]
        The real values are: 
        [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4 9 6 6 5 4 0 7 4 0 1 3 1 3 4 7 2 7
        1 2 1 1 7 4 2 3 5 1 2 4 4]
        Accuracy:  97.92999999999999  %
        The size of each batch:  16
        The total number of epochs:  50
        Neuron layer dimensions : [784 -> 64 -> 10]
        Dataset size: 60000
        learning rate = 0.2
    }


CheckList:

    ===================================== TASKS ====================================  |  +++++++ STATE +++++++    | ------- DATE OF COMPLETION ------- | 
    
    1) Test the learning rate of 0.275 (lr = 0.275 ) in the final testing region      |        unfinished       |                                    |
    2) Check if the number of epochs when changed have a difference with lr = 0.275   |        unfinished       |                                    |


'''

# Starts here!

import numpy as np
import matplotlib.pyplot as plt

def init_params(layer_dims):
    """
    He-style initialization (good default for ReLU networks).
    """
    params = {}
    L = len(layer_dims)
    
    for l in range(1 , L):
        
        params['W' + str(l)] = (
            np.random.randn(
                layer_dims[ l ] , 
                layer_dims[ l - 1 ] 
                ).astype(np.float32) 
                * np.sqrt(2.0 / (layer_dims[ l ] + layer_dims[ l - 1 ]) 
            )
        )
        
        params['b' + str(l)] = np.zeros(
                (
                    layer_dims[l] , 
                    1
                ) , 
                dtype = np.float32
            )
    
    return params

#ReLU activation function
def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A , cache

def relu_backward(dA , Z):
    """
    Derivative of ReLU: 1 where Z > 0, else 0.
    """
    dZ = dA.copy()
    dZ[ Z <= 0 ] = 0
    
    return dZ


#Softmax activation function
def softmax(Z):
    expZ = np.exp(
        Z - np.max(Z , axis = 0 , keepdims = True)
    )
    cache = Z
    return expZ / np.sum(
        expZ , axis = 0 , keepdims = True
    ) , cache


# Now the forward propragation algo using 
# Z (Linear hypothesis) -> Z = WX + b

# For now FORWARD_PROP IS DONE! 

def forward_prop(X , params):
    """
    Forward pass: ReLU in hidden layers, Softmax at the output.
    """
    A = X # input to first input layer
    caches = []
    L = len(params) // 2
    
    for l in range(1 , L+1):
        
        A_prev = A
        
        W = 'W' + str(l)
        b = 'b' + str(l)
        #linear hypothesis (Z vector)
        Z = np.dot( params[ W ] , A_prev) + params[ b ]
        
        #Storing the linear cache
        linear_cache = (A_prev , params[ W ] , params[ b ])
        
        if l == L :
            #Applying the sigmoid on hypothesis Z matrix vector to make it the cache
            #The A vector stores the activated by softmax value
            A , activation_cache = softmax(Z)
        else:
            #Applying the sigmoid on hypothesis Z matrix vector to make it the cache
            #The A vector stores the activated by sigmoid value
            A , activation_cache = relu(Z)
        
        
        #Storing both linear and activation cache
        cache = (linear_cache , activation_cache)
        caches.append(cache)
    
    return A , caches



# New cost function (NOT binary cross entropy , instead categorical cross entropy)
# As the MNIST dataset has 10 classes
def cost_function(A , Y):
    m = Y.shape[1]
    epsilon = 1e-12
    
    #epsilon is a numerical protection
    cost = -(1/m) * np.sum(
        Y * np.log(A + epsilon)
    )
    return cost


#Softmaxxed Output layer backward pass
# A[2] = softmax(Z[2])
# dZ[2] = A[2] - Y

def output_layer_backward(AL , Y , cache):
    linear_cache , activation_cache = cache

    A_prev , W , b = linear_cache
    
    m = A_prev.shape[1]
    
    #differentiated values dZ , dW , db
    
    dZ = AL - Y
    dW = (1/m) * np.dot(dZ , A_prev.T)
    db = (1/m) * np.sum(
        dZ , axis = 1 , keepdims = True
    )
    
    dA_prev = np.dot(W.T , dZ)
    
    return dA_prev , dW , db

#Older sigmoid activation function backpass for hidden layers
#Backpropagation one layer by one layer
def one_layer_backward(dA , cache):
    """
    Backward pass for a ReLU layer.
    """
    linear_cache , activation_cache = cache
    
    Z = activation_cache
    
    dZ = relu_backward(dA , Z) # derivative of the relu function
    
    A_prev , W , b = linear_cache
    m = A_prev.shape[1]
    
    dW = (1/m) * np.dot( dZ , A_prev.T)
    db = (1/m) * np.sum( dZ , axis = 1 , keepdims = True)
    dA_prev = np.dot(W.T , dZ)
    
    return dA_prev , dW , db

#New total backprop for MNIST data entries as pictures
def backprop(AL , Y , caches):
    grads = {}
    L = len(caches)
    
    # -------------------------
    # Output layer
    # -------------------------
    
    current_cache = caches[L - 1]
    
    dA_prev, dW, db = output_layer_backward(
        AL , Y , current_cache
    )
    
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    
    # -------------------------
    # Hidden layers
    # -------------------------
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        
        dA_prev_temp , dW_temp , db_temp = one_layer_backward(
            grads[ "dA" + str(l+1) ] , 
            current_cache)
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads



#Updating the parameters
def update_params(parameters , grads , learning_rate):
    L= len(parameters) // 2
    
    for l in range(L):
        parameters['W'+str(l+1)] =(
            parameters['W'+str(l+1)] - 
            learning_rate*grads['dW'+str(l+1)]
        )
        parameters['b'+str(l+1)] = (
            parameters['b'+str(l+1)] - 
            learning_rate*grads['db'+str(l+1)]
        )
    return parameters


'''
# A train() method that loops over each epoch manually without batch breaking.

# train() method for the training of the network

def train(X, Y, layer_dims, epochs, lr, print_every):

    params = init_params(layer_dims)
    cost_history = []

    for i in range(epochs):

        # 1. Forward propagation
        Y_hat, caches = forward_prop(X, params)

        # 2. Calculate loss
        cost = cost_function(Y_hat, Y)
        cost_history.append(cost)

        # 3. Backpropagation
        grads = backprop(Y_hat, Y, caches)

        # 4. Update parameters
        params = update_params(
            params,
            grads,
            lr
        )

        if i % print_every == 0:
            print(
                "Epoch:", i,
                "| Cost:", cost
            )

    return params, cost_history
'''


'''
# train() method with batch breaking
def train( X , Y , layer_dims , epochs , lr , batch_size , print_every ):
    
    #initialzation
    params = init_params(layer_dims)
    cost_history = []
    
    for epoch in range(epochs):
        # the columns of the X vector is number of training examples 
        # Such as X.shape = (784 , 60000);  X.shape[1] = 60000
        m = X.shape[1]
        
        #initialize the cost of each epoch
        epoch_cost = 0
        
        #permutation for shuffling the combinations
        permutation = np.random.permutation(m)

        for start in range(0, m, batch_size):
            # find the end of the batch to make sure we don't go 
            # beyond the number of given examples
            # as 10000 / 128 = 78 (rem = 16) 
            # last batch is 16 examples.
            # We use min() to safeguard our batch set

            end = min(start + batch_size, m)

            batch_indices = permutation[start:end]

            X_batch = X[:, batch_indices]
            Y_batch = Y[:, batch_indices]
            
            
            # -------------------------
            # Forward propagation
            # -------------------------

            Y_hat, caches = forward_prop(
                X_batch,
                params
            )

            # -------------------------
            # Cost
            # -------------------------

            cost = cost_function(
                Y_hat,
                Y_batch
            )

            epoch_cost += cost * X_batch.shape[1] 

            # -------------------------
            # Backpropagation
            # -------------------------

            grads = backprop(
                Y_hat,
                Y_batch,
                caches
            )

            # -------------------------
            # Update parameters
            # -------------------------

            params = update_params(
                params,
                grads,
                lr
            )


        # -------------------------
        # Average epoch cost
        # -------------------------

        epoch_cost /= m 
        cost_history.append(epoch_cost)
        
        
        # -------------------------
        # Display
        # -------------------------
        if epoch % print_every == 0:
            print(
                "Epoch:", epoch,
                "| Cost:", epoch_cost
            )
            
    return params, cost_history

'''

def train(X, Y, layer_dims, epochs, lr, batch_size, print_every):
    """
    Mini-batch training loop with extensive diagnostic printing
    (the version that was used for the documented runs).
    """
    params = init_params(layer_dims)
    cost_history = []

    for epoch in range(epochs):

        m = X.shape[1]
        epoch_cost = 0

        permutation = np.random.permutation(m)

        for start in range(0, m, batch_size):

            end = min(start + batch_size, m)

            batch_indices = permutation[start:end]

            X_batch = X[:, batch_indices]
            Y_batch = Y[:, batch_indices]

            # ---------------- FORWARD ----------------

            Y_hat, caches = forward_prop(
                X_batch,
                params
            )
            
            # ---------------- COST ----------------

            cost = cost_function(
                Y_hat,
                Y_batch
            )

            epoch_cost += cost * X_batch.shape[1]

            # ---------------- BACKPROP ----------------

            grads = backprop(
                Y_hat,
                Y_batch,
                caches
            )

            # =========================================================
            #                 NETWORK DIAGNOSTICS
            # =========================================================

            if epoch % print_every == 0 and start == 0:

                print("\n")
                print("=" * 75)
                print("NETWORK DIAGNOSTICS")
                print("=" * 75)

                print("Epoch          :", epoch)
                print("Batch          :", start // batch_size + 1)
                print("Learning rate  :", lr)
                print("Batch size     :", X_batch.shape[1])
                print("Cost           :", cost)

                # -----------------------------------------------------
                # PARAMETERS
                # -----------------------------------------------------

                print("\n--- PARAMETERS ---")

                L = len(params) // 2

                for l in range(1, L + 1):

                    W = params["W" + str(l)]
                    b = params["b" + str(l)]

                    print(
                        f"W{l} | "
                        f"shape={W.shape} | "
                        f"min={np.min(W):.6e} | "
                        f"max={np.max(W):.6e} | "
                        f"mean={np.mean(W):.6e} | "
                        f"std={np.std(W):.6e}"
                    )

                    print(
                        f"b{l} | "
                        f"shape={b.shape} | "
                        f"min={np.min(b):.6e} | "
                        f"max={np.max(b):.6e} | "
                        f"mean={np.mean(b):.6e} | "
                        f"std={np.std(b):.6e}"
                    )

                # -----------------------------------------------------
                # GRADIENTS
                # -----------------------------------------------------

                print("\n--- GRADIENTS ---")

                for l in range(1, L + 1):

                    dW = grads["dW" + str(l)]
                    db = grads["db" + str(l)]

                    print(
                        f"dW{l} | "
                        f"min={np.min(dW):.6e} | "
                        f"max={np.max(dW):.6e} | "
                        f"mean={np.mean(dW):.6e} | "
                        f"std={np.std(dW):.6e} | "
                        f"norm={np.linalg.norm(dW):.6e}"
                    )

                    print(
                        f"db{l} | "
                        f"min={np.min(db):.6e} | "
                        f"max={np.max(db):.6e} | "
                        f"mean={np.mean(db):.6e} | "
                        f"std={np.std(db):.6e} | "
                        f"norm={np.linalg.norm(db):.6e}"
                    )

                # -----------------------------------------------------
                # GRADIENT / PARAMETER RATIO
                # -----------------------------------------------------

                print("\n--- UPDATE SCALE ---")

                for l in range(1, L + 1):

                    W = params["W" + str(l)]
                    b = params["b" + str(l)]

                    dW = grads["dW" + str(l)]
                    db = grads["db" + str(l)]

                    W_ratio = (
                        np.linalg.norm(lr * dW) /
                        (np.linalg.norm(W) + 1e-12)
                    )

                    b_ratio = (
                        np.linalg.norm(lr * db) /
                        (np.linalg.norm(b) + 1e-12)
                    )

                    print(
                        f"Layer {l} | "
                        f"ΔW/W = {W_ratio:.6e} | "
                        f"Δb/b = {b_ratio:.6e}"
                    )

                # -----------------------------------------------------
                # NaN / INF CHECK
                # -----------------------------------------------------

                print("\n--- NUMERICAL CHECK ---")

                numerical_problem = False

                for l in range(1, L + 1):

                    W = params["W" + str(l)]
                    b = params["b" + str(l)]

                    dW = grads["dW" + str(l)]
                    db = grads["db" + str(l)]

                    if (
                        not np.all(np.isfinite(W)) or
                        not np.all(np.isfinite(b)) or
                        not np.all(np.isfinite(dW)) or
                        not np.all(np.isfinite(db))
                    ):
                        numerical_problem = True

                if numerical_problem:
                    print("!!! NaN / INF DETECTED !!!")
                else:
                    print("All parameters and gradients are finite.")

                # -----------------------------------------------------
                # OUTPUT CHECK
                # -----------------------------------------------------

                print("\n--- OUTPUT CHECK ---")

                print(
                    "Output min   :",
                    np.min(Y_hat)
                )

                print(
                    "Output max   :",
                    np.max(Y_hat)
                )

                print(
                    "Output mean  :",
                    np.mean(Y_hat)
                )

                print(
                    "Probability sums:",
                    np.sum(Y_hat, axis=0)[:5]
                )

                # -----------------------------------------------------
                # RECOGNITION CHECK
                # -----------------------------------------------------

                predictions = np.argmax(Y_hat, axis=0)
                actual = np.argmax(Y_batch, axis=0)

                batch_accuracy = np.mean(
                    predictions == actual
                ) * 100

                print(
                    "\nBatch accuracy :",
                    batch_accuracy,
                    "%"
                )

                print("=" * 75)

            # ---------------- UPDATE ----------------

            params = update_params(
                params,
                grads,
                lr
            )

        # =============================================================
        #                  END OF EPOCH
        # =============================================================

        # Correct weighted-average cost
        epoch_cost /= m

        cost_history.append(epoch_cost)

        if epoch % print_every == 0:

            print(
                f"\nEpoch {epoch} COMPLETE"
                f" | Average Cost: {epoch_cost:.10f}"
            )

    return params, cost_history

def predict(X, params):

    # Run the image(s) through the network
    AL, caches = forward_prop(X, params)

    # Find the output neuron with the highest probability
    predictions = np.argmax(AL, axis=0)

    return predictions, AL


'''
X_fake = np.random.rand(784, 5)

Y_fake = np.eye(10)[:, [0, 3, 7, 2, 9]]

params = init_params([784, 64, 10])

AL, caches = forward_prop(X_fake, params)

grads = backprop(AL, Y_fake, caches)

print("AL:", AL.shape)
print("dW1:", grads["dW1"].shape)
print("db1:", grads["db1"].shape)
print("dW2:", grads["dW2"].shape)
print("db2:", grads["db2"].shape)
'''

# -----------------------------
# SMALL TEST
# -----------------------------

X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")
y_test = np.load("y_test.npy")

testValues = 60000
X_demo = X_train[:, :testValues]
Y_demo = Y_train[:, :testValues]

batchsize = 16
totalEpochs = 50

learning_rate = 0.1

params, cost_history = train(
    X_demo,
    Y_demo,
    [ 784, 64, 10 ],
    epochs=totalEpochs,
    lr=learning_rate, 
    batch_size = batchsize, 
    print_every=10
)

predictions , probabilities = predict(X_test , params)
print("The guess values are: \n",predictions[:50])

print("The real values are: \n" , y_test[:50])

accuracy = np.mean(predictions == y_test) * 100

print("Accuracy: " , accuracy , " %")

print("The size of each batch: " , batchsize)
print("The total number of epochs: " , totalEpochs)


plt.plot(cost_history)

plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.yscale("log") # logarithmic scale graph

plt.title("Training Cost (see VScode terminal for data during training)")

plt.get_current_fig_manager().set_window_title( # type: ignore
    "Cost(y) vs Epochs(x) graph for the simple data set (see vscode terminal)"
)
plt.show()
