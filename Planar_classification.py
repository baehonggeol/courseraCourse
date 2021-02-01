#Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Planar_util import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets



np.random.seed(1) # set a seed so that the results are consistent

# 꽃 데이터를 가지고 온다
X, Y = load_planar_dataset()


# Visualize the data:
# 분포도로 데이터 시각화 하기
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)

# 입력값 크기와 출력값 크기
shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  # training set size # 트레이닝 데이터 사이즈


print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

# Train the logistic regression classifier
# 간단한 선형 모델을 이용한 학습
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

# Plot the decision boundary for logistic regression
# 모델을 기반으로 나눈다.
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
# 정확도를 측정한다.
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")


# 각 hidden layer 의 사이즈를 설정한다.
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """

    n_x = X.shape[0]  # size of input layer # 입력값에 비례한 사이즈 행의 크기를 말한다.
    n_h = 4 # 이건 언제나 바뀔수 있고 하나의 hyper parameter로 사용하여 모델의 깊이와 성능을 향상 시킬수 있다.
    n_y = Y.shape[0]  # size of output layer # 출력값은 당연히 내가 원하는 크기 값이 나와야한다.

    return (n_x, n_h, n_y)


X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

    # random으로 시작하는 이유는 다름 아닌 weight가 0으로 시작할 경우 첫 학습의 답은  0이기 때문에 손해를 보게된다
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x)) # 이 모양은 변하지 말아야한다.
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

n_x, n_h, n_y = initialize_parameters_test_case()

parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    # 초기화 한 가중치 값을 dict에서 불러온다.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    #  yes or no를 가릴땐 sigmoid가 쓰이지만 그외는 tanh가 더 좋은 성능을 발휘한다. 그래서 마지막에만 sigmoid를 쓰긴한다.
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)


    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)

# Note: we use the mean here just to make sure that your output matches ours.
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    [Note that the parameters argument is not used in this function,
    but the auto-grader currently expects this parameter.
    Future version of this notebook will fix both the notebook
    and the auto-grader so that `parameters` is not needed.
    For now, please include `parameters` in the function signature,
    and also when invoking this function.]

    Returns:
    cost -- cross-entropy cost given equation (13)

    """

    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost

    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y) #loss 값 formula
    # logprobs = np.multiply(Y,np.log(A2))+np.multiply(1-Y,np.log(1-A2))
    cost = -1 / m * np.sum(logprobs)


    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17
    assert (isinstance(cost, float))

    return cost

A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".

    W1 = parameters["W1"] #레이어 1번의 가중치
    W2 = parameters["W2"] # 레이어 2번의 가중치

    # Retrieve also A1 and A2 from dictionary "cache".

    A1 = cache["A1"] # forward propagation을 통해 나온 첫번째 레이어의 A값을 가지고 온다
    A2 = cache["A2"] # forward propagation을 통해 나온 두번째 레이어의 A값을 가지고 온다


    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = cache["A2"] - Y #실제 값과 차이를 구한다.
    dW2 = 1 / m * np.dot(dZ2, cache["A1"].T) # 그 차이를 기준으로 미분한다. #두번째 레이어 구한다.
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True) # 바이어스 값 또한 업데이트 시킨다.  #두번째 레이어 구한다
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2)) # 첫번째 레이어 A값 구한다
    dW1 = 1 / m * np.dot(dZ1, X.T) # 첫번째 레이어 가중치 구한다
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)# 첫번째 레이어 바이어스 구한다

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    # Retrieve each parameter from the dictionary "parameters"
    # 초기의 forward propagation을 통해 구한 값을 불러온다
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    # 미분한 값들을 불러온다
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    # 학습률(learning rate)의 크기의 따라 변화량을 정한다.
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    # 위에 계산된 가중치와 바이어스를 기존 가중치와 바이어스를 덮어쓴다.
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))




#모델링
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):


    """

    Arguments: # 입력값들
    X -- dataset of shape (2, number of examples) 입렫데이터
    Y -- labels of shape (1, number of examples) 출력 데이터
    n_h -- size of the hidden layer 레이어의 갯수
    num_iterations -- Number of iterations in gradient descent loop 모델 훈련 횟수
    print_cost -- if True, print the cost every 1000 iterations 1000번마다 cost값 출력하기

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.  출력값은 가중치와 바이어스로 이루어진 모델
    """

    np.random.seed(3)
    # 입력 값의 shape을 맞추어 준다.
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters
    # 지정된 shape에 따라 가중치와 바이어스를 초기화 한다.
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Loop (gradient descent)
    #원하는 훈련 횟수만큼 돌린다.

    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        # 앞전파 하여 A2 마지막 값과 레이어 안에 있는 값들은 cache안에 저장
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        # 앞전파의 cost값 산정
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        # 뒷전파를 해서 미분하여 각도를 계산한다.
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        # 학습률의 따라 모든 가중치와 바이어스를 바꾸어준다.
        parameters = update_parameters(parameters, grads)


        # Print the cost every 1000 iterations
        # 매 천번당 cost값을 출력
        if print_cost and i % 1000 == 0: # 1000으로 나누었을때 0이면 true
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    # 바이너리 모델이기 때문에 확률의 따라 맞다와 아니다로 답을한다.
    A2, cache = forward_propagation(X, parameters) # 모델의 있는 값을 사용하여 새로 들어오는 X 입력값을 앞전파한다.
    # predictions = (A2 > 0.5)
    # 50% 보다 더 높게 잡아도 된다. 여기선 50% 확률을 잡았다.
    predictions = (A2 > 0.5)

    return predictions

parameters, X_assess = predict_test_case()


predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True) # 레이어가 4개인 모델을 만들며 10000번의 훈련을 한다.

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))


# Print accuracy
# 정확도 측정
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))


# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}


dataset = "gaussian_quantiles"


X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y%2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);