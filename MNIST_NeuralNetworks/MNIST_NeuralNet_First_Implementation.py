'''
Code Property of: Asinak Goswami (GitHub username: asinakgoswami-hue)

Date of push: 16/11/2026 (DD/MM/YYYY)

This is a learning implementation. Feel free to add comments on what I should work on next!

'''

# ======================================== ACTUAL CODE ========================================

"""
MNIST_Softmax_Sigmoid_NeuralNet.py
----------------------------------
Fully-connected network for MNIST using Sigmoid activations in the hidden layers
and Softmax + categorical cross-entropy at the output.

This is one of the intermediate stages of the project (before switching the
hidden layers to ReLU).

All original commented-out experiments and older train loops are kept.
"""

import numpy as np
import matplotlib.pyplot as plt

'''
* The first 90% of this network was reached on 24 / 9 / 26
* The first 93% of this network was reached on 24 / 9 / 26
* Tasks: 
    - To improve the accuracy of the network to better standards
    - To implement more neurons such as a 64 x 64 grid from a picture of my own handwriting
    - To make a [4096 , 128 , 64 , 32 , 10] network to read and guess my own handwriting.
    - Build standalone image → prediction pipeline
    
LOGS ::::

Experiment | Architecture |HL Activation | Data | Batch | Epochs | LR | Accuracy | Date
------------------------------------------------------------------------------------------------
01         | 784→64→10     | Sigmoid    | 10k  | 16    | 20       | .1 | 93.70% | 24 / 9 / 26
02         | 784→128→64→10 | Sigmoid    | 10k  | 16    | 30       | .1 | 96.59% | 24 / 9 / 26
03         | 784→64→10     | ReLU       | 15k  | 16    | 50       | .1 | 96.63% | 24 / 9 / 26
04         | 784→64→10     | ReLU       | 60k  | 16    | 50       | .1 | 98.06% | 24 / 9 / 26
05         | 784→64→10     | Sigmoid    | 60k  | 16    | 50       | .1 | 97.68% | 24 / 9 / 26

'''

def init_params(layer_dims):
    """
    Initialize weights with a simple scaled normal distribution
    and biases to zero.
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
        
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return params

#Sigmoid function sigma(z1)

def sigmoid(Z): 
    """
    Classic sigmoid. Returns activated values + the original Z for the backward pass.
    """
    A = 1 / ( 1 + np.exp( np.dot(-1,Z) ) )
    
    cache = (Z)
    
    return A , cache


# Now the forward propragation algo using 
# Z (Linear hypothesis) -> Z = WX + b

# For now FORWARD_PROP IS DONE! 

def forward_prop(X , params):
    """
    Full forward pass.
    Hidden layers → sigmoid, output layer → softmax.
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
            A , activation_cache = sigmoid(Z)
        
        
        #Storing both linear and activation cache
        cache = (linear_cache , activation_cache)
        caches.append(cache)
    
    return A , caches


#Softmax activation function
def softmax(Z):
    """
    Numerically stable softmax (subtract max before exp).
    """
    expZ = np.exp(
        Z - np.max(Z , axis = 0 , keepdims = True)
    )
    cache = Z
    return expZ / np.sum(
        expZ , axis = 0 , keepdims = True
    ) , cache
    
# New cost function (NOT binary cross entropy , instead categorical cross entropy)
# As the MNIST dataset has 10 classes
def cost_function(A , Y):
    """
    Categorical cross-entropy.
    A and Y both have shape (10, batch_size).
    """
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
    """
    Gradient of Softmax + categorical cross-entropy is simply (AL - Y).
    """
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
    Backward pass for a single sigmoid layer.
    """
    linear_cache , activation_cache = cache
    
    Z = activation_cache
    A = sigmoid(Z)[0] 
    
    dZ = dA * A * (1 - A) # derivative of the sigmoid function
    
    A_prev , W , b = linear_cache
    m = A_prev.shape[1]
    
    dW = (1/m) * np.dot( dZ , A_prev.T)
    db = (1/m) * np.sum( dZ , axis = 1 , keepdims = True)
    dA_prev = np.dot(W.T , dZ)
    
    return dA_prev , dW , db

#New total backprop for MNIST data entries as pictures
def backprop(AL , Y , caches):
    """
    Full backpropagation through all layers.
    """
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
    """
    Classic SGD update.
    """
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

# train() method with batch breaking
def train( X , Y , layer_dims , epochs , lr , batch_size , print_every ):
    """
    Mini-batch SGD training loop.
    """
    #initialzation
    params = init_params(layer_dims)
    cost_history = []
    
    for epoch in range(epochs):
        # the columns of the X vector are the training examples
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

        # FIXED: original code divided by number of batches instead of by m.
        # That made the reported cost depend on the chosen batch size.
        # Correct weighted average is simply total_cost / m.
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



def predict(X, params):
    """
    Run inference and return both the predicted class indices and the full probability matrix.
    """
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

params, cost_history = train(
    X_demo,
    Y_demo,
    [ 784, 64, 10 ],
    epochs=totalEpochs,
    lr=0.1, 
    batch_size = batchsize, 
    print_every=10
)

predictions , probabilities = predict(X_test , params)
print("The guess values are: \n",predictions[:10])

print("The real values are: \n" , y_test[:10])

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
