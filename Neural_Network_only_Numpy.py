import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
#데이터 불러 오기
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
#25번째 사진을 본다
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")


#데이터 크기와 형태 확인하기
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig[0].shape[0]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

'''
train_set_x_flatten shape: (12288, 209)
train_set_y shape: (1, 209)
test_set_x_flatten shape: (12288, 50)
test_set_y shape: (1, 50)
sanity check after reshaping: [17 31 56 22 33]
'''



# Reshape the training and test examples
#.T는 transpose로 뒤집어 어
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))



# 킬라 이미지 크기는 0~255 이를 0~1로 변환을 한다 이를 standardation이라고 한다.
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


#시그모이드 함수는 모 아니면 도인 상황에서 많이 쓰인다. 값은 0~1 사이로 값이 나온다.
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###

    return s

print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))


# 가중치와 bia값을 0으로 초기화 하여 저장 공간을 확보한다.
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros((dim, 1))
    b = 0

    return w, b

# 여기서 dim은 레이어의 사이즈를 말한다. 2개 뉴런의 1개 레이어를 만드는것이다.
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1] #이미지 사이즈

    A = sigmoid(np.dot(w.T, X) + b) #  시그모이드 함수를 통해 A값을 구한다
    cost = -1 / m * (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))  # compute cost 값을 구한다.

    # BACKWARD PROPAGATION (TO FIND GRAD)
    # 실제 값과 A값을 빼서 입력값에 곱한다. 그 차이만큼 가중치를 조정해 들어간다.
    # 바이어스값도 똑같이 진행된다.

    dw = 1 / m * (np.dot(X, ((A - Y).T)))
    db = 1 / m * (np.sum(A - Y))

    cost = np.squeeze(cost)
    #디션
    grads = {"dw": dw,
             "db": db}

    return grads, cost

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))


#이곳에서 forward propagation과 backpropagation을 횟수 = iter 와  학습률을 조정하여 여러번 학습 시킨다.
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    #나중에 cost 값을 그래프에 표기용으로 모와서 문제점을 진단하는데 사용한다.
    costs = []

    #반복문으로 지정한 횟수 만큼 훈련 시킨다.
    for i in range(num_iterations):

        # Cost and gradient calculation
        #cost값과 A값을 구한다.

        grads, cost = propagate(w, b, X, Y)


        # Retrieve derivatives from grads
        # backpropagation 가중치 값들을 저장소 넣고
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        # 학률에 따라 조정한다.
        w = w - learning_rate * dw
        b = b - learning_rate * db


        # Record the costs
        # 100번 마다 한번씩 cost값을 넣는다.
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        # 100번 마다 cost값을 프린트한다.
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

'''
w = [[ 0.19033591]
 [ 0.12259159]]
b = 1.92535983008
dw = [[ 0.67752042]
 [ 1.41625495]]
db = 0.219194504541
'''


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1] # 3
    Y_prediction = np.zeros((1, m)) # 1,3 np.array([0,0,0])
    w = w.reshape(X.shape[0], 1) # 2,1

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b) #1,3

    for i in range(A.shape[1]): # 3번의 반복을 하며
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        Y_prediction[0][i] = 1 if A[0][i] > 0.5 else 0 # 예측값을 0.5만 넘으면 1로 변환

    return Y_prediction

w = np.array([[0.1124579],[0.23106775]]) # 2,1
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]]) #2,3
print ("predictions = " + str(predict(w, b, X)))

#predictions = [[ 1.  1.  0.]]

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # initialize parameters with zeros
    #입력값 크기에 맞게 가중치값 크기를 설정한다.
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    # 각도 값을 구하며 cost 값을 구한다. 횟수만큼 반복하여 cost값을 줄어간다.
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)

    # Retrieve parameters w and b from dictionary "parameters"
    # parameter dict에서 값을 가지고 온다.
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test) # test 값에 사용해서
    Y_prediction_train = predict(w, b, X_train) # train 값에 사용해서


    # Print train/test Errors
    # 트레이닝 데이터로 학습한 결과와 그 결과를 토대로 한번도 안본 test데이터의 결과를 비교하여 그 성능을 향상 시킬 에러 분석이 들어가야 한다.
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    #이 모든걸 저장하고 리턴 한다.
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
'''
이러한 차이는 오버핏을 의심 해야 하고 데이터가 아직 부족할수도 있다. 극심한 경우 트레이닝 데이터와 테스트 데이터 간에 데이터의 차이가 심한 경우도 있다.
train accuracy: 99.04306220095694 %
test accuracy: 70.0 %
'''


# Example of a picture that was wrongly classified.
# 잘못 분류된 사진
index = 9
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")



# Plot learning curve (with costs)
#cost값 그래프 배치를 사용하지 않는 이상 부드럽게 내려가야한다.
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# 학습률을 여러가지를 사용하여 진행해 본다. 학습률을 줄이게 되면 더욱 차이를 근소 하게 만든다. 대신 오래 걸린다. 그리고 너무 낮으면 글로벌 버텀이 아닌 로컬 버텀에서 멈출수 있다.
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


my_image = "my_image.jpg" #나만의 이미지를 넣어서 실력을 확인해본다.


# We preprocess the image to fit your algorithm.
fname = "images/" + my_image #path
image = np.array(ndimage.imread(fname, flatten=False))
image = image/255. #standardation하고
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T #입력값 크기로 맞춘다. 이것을 데이터 preprocessing이라고 한다.
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")