import numpy as np
from datetime import datetime

# 활성화 함수(activative function): sigmoid 함수
# 0 ~ 1 사이의 값을 가지도록 한다.
def sigmoid(x):
   return  np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))

# 수치 미분
def numerical_derivative(f, x):
    """
    수학적인 lim(x->0)을 프로그램으로 구현하기 위해서 아주 작은 값으로 설정.
    컴퓨터는 32bit나 64bit 등 한정된 크기로만 수를 나타낼 수 있으므로 lim 개념을
    정확하게 나타내지는 못한다. = 수치 미분
    """
    delta_x = 1e-4  # 연구를 통해 10에 -4승이 가장 잘 나타낸다고 알려짐
    grad = np.zeros_like(x)  # x의 크기를 가지는 리스트를 만들어서 모두 0으로 초기화
    # iterator를 정의하여 x의 처음부터 끝까지 순서대로 지나간다.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:    # iterator로 x의 끝이 아닐 때까지 반복
        idx = it.multi_index  # it가 가리키는 값의 index를 idx에 저장
        tmp_val = x[idx]      # x[idx]의 값이 아래의 계산 도중에 바뀔 수 있으므로 임시 저장

        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2*delta_x)  # 미분 공식 구현
        x[idx] = tmp_val  # 바뀐 x 값에 원래 값 다시 저장
        it.iternext()  # 다음 순서로 넘어가기

    return grad  # 최종적으로 각 변수에 대한 편미분 결과를 return

# NN Class
class NeuralNetwork:

    # 생성자
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        # 각 층의 개수 지정
        # input = 784, hidden = 100, output = 10
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        """
        2층 hidden layer unit
        임의로 은닉층의 노드 개수는 100개로 지정함
        가중치 W, 바이어스 b 초기화
        W2 = (784, 100)
        b2 = (100, )
        """
        # 가중치를 Xavier/He 방법으로 초기화
        self.W2 = np.random.randn(self.input_nodes, self.hidden_nodes) / np.sqrt(self.input_nodes/2)
        self.b2 = np.random.rand(self.hidden_nodes)

        """
        3층 output layer unit
        결과는 1~9 사이의 숫자를 가지므로 출력층의 노드 수는 10개
        가중치 W, 바이어스 b 초기화
        W3 = (100, 10)
        b3 = (10, )
        """
        self.W3 = np.random.randn(self.hidden_nodes, self.output_nodes) / np.sqrt(self.hidden_nodes/2)
        self.b3 = np. random.rand(self.output_nodes)

        # 학습률 초기화
        self.learning_rate = learning_rate

        print("NeuralNetwork object is created!")

    # feed forward를 이용하여 입력층에서 출력층까지 데이터를 전달하고 손실 함수 값 계산
    # loss_val(self) method와 동일한 코드이지만 loss_val(self)는 외부 출력용으로 사용되는 것이고,
    # feed_forward(self)는 학습을 위해 사용되는 method
    def feed_forward(self):
        delta = 1e-7  # log 무한대 발산을 막기 위해
        
        # 입력층-은닉층
        z2 = np.dot(self.input_nodes, self.W2) + self.b2
        a2 = sigmoid(z2)

        # 은닉층-출력층
        z3 = np.dot(a2, self.W3) + self.b3
        y = a3 = sigmoid(z3)

        # Cross-entropy
        # logistic Regression-Classification에서의 손실 함수
        return -np.sum( self.target_data*np.log(y+delta) + (1-self.target_data)*np.log((1-y)+delta) )

    # 외부 출력을 위한 loss_val(self)
    def loss_val(self):
        delta = 1e-7

        z2 = np.dot(self.input_nodes, self.W2) + self.b2
        a2 = sigmoid(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        y = sigmoid(z3)

        return -np.sum( self.target_data*np.log(y+delta) + (1-self.target_data)*np.log((1-y)+delta) )

    # 수치미분을 이용하여 손실함수가 최소가 될 때까지 학습하는 함수
    def train(self, input_data, target_data):
        # input_data와 target_data 초기화
        self.input_data = input_data
        self.target_data = target_data
        
        # 손실함수 정의
        f = lambda x : self.feed_forward()

        # 편미분(수치미분)을 통해 W와 b를 update    
        self.W2 -= self.learning_rate * numerical_derivative(f, self.W2)
        self.b2 -= self.learning_rate * numerical_derivative(f, self.b2)
        self.W3 -= self.learning_rate * numerical_derivative(f, self.W3)
        self.b3 -= self.learning_rate * numerical_derivative(f, self.b3)

    # query, 미래 값 예측 함수
    def predict(self, input_data):
        z2 = np.dot(input_data, self.W2) + self.b2
        a2 = sigmoid(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        y = sigmoid(z3)

        # MNIST의 경우는 one-hot encoding을 적용하기 때문에
        # 0 또는 1이 아닌 argmax()를 통해 최대 인덱스를 넘겨줘야 함
        predicted_num = np.argmax(y)

        return predicted_num

    # 정확도 측정 함수
    def accuracy(self, input_data, target_data):

        matched_list = []
        not_matched_list = []

        # list which contains (index, label, prediction) value
        index_label_prediction_list = []
        
        # temp list which contains label and prediction in sequence
        temp_list = []
        
        for index in range(len(input_data)):
            
            # test_data의 1열에 있는 정답을 분리
            label = int(target_data[index])
                        
            # normalize
            data = (input_data[index, :] / 255.0 * 0.99) + 0.01
      
            predicted_num = self.predict(data)

            # 정답과 예측 값이 맞으면 matched_list에 추가
            if label == predicted_num:
                matched_list.append(index)
                
            else:  # 틀리면 not_matched_list에 추가
                not_matched_list.append(index)
                
                temp_list.append(index)
                temp_list.append(label)
                temp_list.append(predicted_num)
                
                index_label_prediction_list.append(temp_list)
                
                temp_list = []
                
        print("Current Accuracy = ", len(matched_list)/(len(input_data)), "%" )
        
        return matched_list, not_matched_list, index_label_prediction_list



# 사용
training_data = np.loadtxt('C:/Users/user/Desktop/work/neowizard/MachineLearning/mnist_train.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('C:/Users/user/Desktop/work/neowizard/MachineLearning/mnist_test.csv', delimiter=',', dtype=np.float32)

print("training data shpae = ", training_data.shape, "test data shpae = ", test_data.shape)

# Hyper-parameter
input_nodes = training_data.shape[1] - 1
hidden_nodes = 30
output_nodes = 10
learning_rate = 1e-2
epochs = 1   # 전체 반복 횟수

# 손실함수 값을 저장할 list 생성
loss_val_list = []

# NeuralNetwork 객체 생성
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print("Neural Network Learning using Numerical Derivative...")

start_time = datetime.now()

for step in range(epochs):

    for index in range(len(training_data)):
        # input_data와 target_data를 0~1 사이의 값으로 정규화(normalize)

        """
        검은색:0, 하얀색:1
        즉, 입력 값은 0~255 사이의 값을 가진다. 이 값을 그대로 사용한다면 손실함수를 계산할 때,
        cross-entropy의 log부분에서 overflow가 발생할 가능성이 매우 높다.
        따라서, 프로그램으로 구현할 때에는 모든 입력 값을 0~1사이의 값으로 만들어주는 정규화를 작업 수행.
        아래는 정규화를 해주는 많은 방법 중 한 가지이다.
        데이터가 가지는 최댓값으로 나누어주어 모든 데이터가 0과 1사이의 값을 가지도록 한다.
        """
        input_data = ((training_data[index, 1:] / 255.0) * 0.99) + 0.01

        # one-hot encoding을 구현하기 위해서 10개의 노드를 모두 0.01로 초기화하고,
        # 정답을 나타내는 인덱스 노드에 가장 큰 값인 0.99로 초기화
        target_data = np.zeros(output_nodes) + 0.01
        target_data[int(training_data[index, 0])] = 0.99

        nn.train(input_data, target_data)

        if(index % 200 == 0):
            print("epochs = ", step, ", index = ", index, ", loss value = ", nn.loss_val())

        # 손실함수 값 저장
        loss_val_list.append(nn.loss_val())

end_time = datetime.now()
print("\nElapsed Time = ", end_time - start_time)

test_input_data = test_data[ :, 1: ] 
test_target_data = test_data[ :, 0 ]

(true_list_1, false_list_1, index_label_prediction_list) = nn.accuracy(test_input_data, test_target_data)