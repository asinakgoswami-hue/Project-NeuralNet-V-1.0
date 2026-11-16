'''
Code Property of: Asinak Goswami (GitHub username: asinakgoswami-hue)

Date of push: 16/11/2026 (DD/MM/YYYY)

This is a learning implementation. Feel free to add comments on what I should work on next!

'''

# ======================================== ACTUAL CODE ========================================

'''

Future Tasks:
1 - to implement a better activation function. NOT SIGMOID!
2 - to slowly graduate this from binary data sets to number from images
3 - use the MNIST dataset to train this bad boy.
4 - learning pytorch to make my life easier.
5 - gradually move towards the linkage in this order:
        C[data] -> C++[computation] -> Java[arrangement] -> Python[execution] -> (NeuralNet)

'''

import numpy as np
import matplotlib.pyplot as plt


def init_params(layer_dims):
    """
    Initialize weights and biases for a fully-connected network.
    layer_dims example: [2, 4, 1]  →  input dim 2, one hidden layer of 4, output dim 1
    """
    params = {}
    L = len(layer_dims)
    
    for l in range(1 , L):
        
        # Simple random initialization (scaled by 0.5). Later networks use better schemes (He / Xavier).
        params['W' + str(l)] = np.random.randn(layer_dims[l] , layer_dims[l-1] ) * 0.5
        
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return params

#Sigmoid function sigma(z1)

def sigmoid(Z): 
    """
    Classic sigmoid activation.
    Returns both the activated values and the original Z (needed for the backward pass).
    """
    A = 1 / ( 1 + np.exp( np.dot(-1,Z) ) )
    
    cache = (Z)
    
    return A , cache

# Now the forward propragation algo using 
# Z (Linear hypothesis) -> Z = WX + b

def forward_prop(X , params):
    """
    Forward pass through the whole network.
    Returns the final activations (predictions) and a list of caches
    (one cache per layer) that are needed for backpropagation.
    """
    A = X # input to first input layer
    caches = []
    L = len(params) // 2
    
    for l in range(1 , L+1):
        
        A_prev = A
        
        #linear hypothesis (Z vector)
        Z = np.dot( params[ 'W' + str(l) ] , A_prev) + params['b' + str(l)]
        
        #Storing the linear cache
        linear_cache = (A_prev , params['W' + str(l)] , params['b' + str(l)])
        
        #Applying the sigmoid on hypothesis Z matrix vector 
        A , activation_cache = sigmoid(Z)
        
        #Storing both linear and activation cache
        cache = (linear_cache , activation_cache)
        caches.append(cache)
    
    return A , caches

#Cost function for the network's forward prop
def cost_function(A , Y):
    """
    Binary cross-entropy loss.
    A = network predictions, Y = ground-truth labels (both shape (1, m) for binary problems).
    """
    m = Y.shape[1]
    
    #Cross entropy loss over a dataset!
    cost = (-1/m) * np.sum(
        Y * np.log(A)
        + 
        (1-Y) * np.log(1-A) 
    )
    
    return cost

#Backpropagation one layer by one layer
def one_layer_backward(dA , cache):
    """
    Backward pass for a single sigmoid layer.
    dA  = gradient coming from the layer above
    cache = (linear_cache, activation_cache) saved during the forward pass
    """
    linear_cache , activation_cache = cache
    
    Z = activation_cache
    A = sigmoid(Z)[0] 
    
    # derivative of the sigmoid function: σ'(z) = σ(z) * (1 - σ(z))
    dZ = dA * A * (1 - A) 
    
    A_prev , W , b = linear_cache
    m = A_prev.shape[1]
    
    dW = (1/m) * np.dot( dZ , A_prev.T)
    db = (1/m) * np.sum( dZ , axis = 1 , keepdims = True)
    dA_prev = np.dot(W.T , dZ)
    
    return dA_prev , dW , db

#Total backprop
def backprop(AL , Y , caches):
    """
    Full backpropagation through all layers.
    Starts from the output layer and works backwards.
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Gradient of the binary cross-entropy loss w.r.t. the final activations
    dAL = - (
        np.divide(Y ,AL) - np.divide(1-Y , 1-AL)
    )
    
    current_cache = caches[L - 1]
    
    dA_prev, dW, db = one_layer_backward(
        dAL,
        current_cache
    )
    
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        
        dA_prev_temp , dW_temp , db_temp = one_layer_backward(grads["dA" + str(l+1)] , current_cache)
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads

def update_params(parameters , grads , learning_rate):
    """
    Simple SGD parameter update:  θ ← θ − lr * dθ
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

#Training function for the network!
def train(X , Y , layer_dims , epochs , lr , print_every):
    """
    Full training loop (no mini-batches – uses the whole dataset every epoch).
    Returns the final parameters and the list of costs for plotting.
    """
    params = init_params(layer_dims)
    cost_history = []
    
    for i in range(epochs):
        
        #---------FORWARD-------------
        Y_hat , caches = forward_prop(X,params)
        
        #---------COST-------------
        cost = cost_function(Y_hat , Y)
        cost_history.append(cost)
        
        #---------BACKPROP-------------
        grads = backprop(Y_hat , Y , caches)
        
        #---------UPDATE-------------
        params = update_params(params , grads , lr)
        
        #---------PRINT TRAINING DATA-------------
        if i % print_every == 0:
            print("\n==============================")
            print("Epoch:", i)
            print("Cost:", cost)

            print("\nY_hat:")
            print(Y_hat)

            print("\nW1:")
            print(params["W1"])

            print("\nb1:")
            print(params["b1"])

            print("\nW2:")
            print(params["W2"])

            print("\nb2:")
            print(params["b2"])
            
            print("dW1:")
            print(grads["dW1"])

            print("db1:")
            print(grads["db1"])

            print("dW2:")
            print(grads["dW2"])

            print("db2:")
            print(grads["db2"])
            
            print("\nGradient magnitudes:")
            print("||dW1|| =", np.linalg.norm(grads["dW1"]))
            print("||db1|| =", np.linalg.norm(grads["db1"]))
            print("||dW2|| =", np.linalg.norm(grads["dW2"]))
            print("||db2|| =", np.linalg.norm(grads["db2"]))
            
            print(
                "LR:", lr,
                "| Final cost:", cost_history[-1],
            )
    return params , cost_history

def test_network(X, Y, params):
    """
    Quick evaluation helper – prints predictions vs ground truth and accuracy.
    """
    print("==================EXPECTED OUTPUTS==================")
    
    print("The input set: \n" , X)
    print("The expected output set: " , Y)    
    
    Y_hat, _ = forward_prop(X, params)

    predictions = (Y_hat > 0.5).astype(int)

    accuracy = np.mean(predictions == Y) * 100
    cost = cost_function(Y_hat, Y)

    print("Predictions:")
    print(Y_hat)

    print("\nBinary predictions:")
    print(predictions)

    print("\nExpected:")
    print(Y)

    print("\nCost:", cost)
    print("Accuracy:", accuracy, "%")





'''
#XOR set
X = np.array([
    [0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1],
    [0,1,0,1, 0,1,0,1, 0,1,0,1, 0,1,0,1]
])
Y = np.array([
    [0,1,1,0, 0,1,1,0, 0,1,1,0, 0,1,1,0]
])
'''


'''
#AND set
X = np.array([
    [0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1],
    [0,1,0,1, 0,1,0,1, 0,1,0,1, 0,1,0,1]
])
Y = np.array([
    [1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1]
])
'''

def main() -> None:
    #NOR set
    X = np.array([
        [0,1,1,1, 0,0,1,1, 0,1,1,1, 0,0,0,1],
        [0,1,0,0, 0,1,0,1, 1,1,1,1, 0,1,0,1]
    ])
    Y = np.array([
        [0,0,1,1, 0,1,1,0, 1,0,0,0, 0,1,0,0]
    ])


    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    for i in range(X.shape[1]):
        print(
            "Input:", X[:, i],
            "Expected:", Y[:, i]
        )

    layer_dims = [2,4,1]

    #Testing learning rates
    lr2 = 0.69314718056
    lr3 = 0.5

    params, cost_history = train(
        X, Y,
        [2, 4, 1],
        epochs=100000,
        lr=lr3,
        print_every=100000
    )

    test_network(X , Y , params)

    #plot for cost vs epochs
    plt.plot(cost_history)

    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.yscale("log") # logarithmic scale graph

    plt.title("Training Cost (see VScode terminal for data during training)")

    plt.get_current_fig_manager().set_window_title( #type: ignore
        "Cost(y) vs Epochs(x) graph for the simple data set (see vscode terminal)"
    )

    plt.show()
    
if __name__ == "__main__":
    main()
