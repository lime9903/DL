import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2*delta_x)
        x[idx] = tmp_val
        it.iternext()
    
    return grad

class LogicGate:

    def __init__(self, gate_name, xdata, tdata):
        
        self.name = gate_name

        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)
        self.__W = np.random.rand(2, 1)
        self.__b = np.random.rand(1)

        self.__learning_rate = 1e-2
    
    def loss_func(self):
        delta = 1e-7

        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)

        return -np.sum( self.__tdata*np.log(y+delta) + (1-self.__tdata)*np.log((1-y)+delta) )

    def error_val(self):
        delta = 1e-7

        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)

        return -np.sum( self.__tdata*np.log(y + delta) + (1-self.__tdata)*np.log( (1-y)+delta ) )

    def train(self):
        f = lambda x : self.loss_func()

        print("[Initial]\n")
        print("error value = ", self.error_val(), "W = ", self.__W, "b = ", self.__b)
        
        print("\n[Process]")
        for step in range(8001):
            self.__W -= self.__learning_rate * numerical_derivative(f, self.__W)
            self.__b -= self.__learning_rate * numerical_derivative(f, self.__b)

            if (step%400==0):
                print("step = ", step, "error value = ", self.error_val(), "W = ", self.__W, "b = ", self.__b)

    def predict(self, input_data):

        z = np.dot(input_data, self.__W) + self.__b
        y = sigmoid(z)

        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result

# USAGE
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
AND_tdata = np.array([0, 0, 0, 1])
OR_tdata = np.array([0, 1, 1, 1])
NAND_tdata = np.array([1, 1, 1, 0])
XOR_tdata = np.array([0, 1, 1, 0])
test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

# AND Gate prediction
AND_obj = LogicGate("AND_GATE", xdata, AND_tdata)
AND_obj.train()

print(AND_obj.name, "\n")
for input_data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(input_data)
    print(input_data, " = ", logical_val, "\n")

# OR Gate prediction
OR_obj = LogicGate("OR_GATE", xdata, OR_tdata)
OR_obj.train()

print(OR_obj.name, "\n")
for input_data in test_data:
    (sigmoid_val, logical_val) = OR_obj.predict(input_data)
    print(input_data, " = ", logical_val, "\n")

#NAND Gate prediction
NAND_obj = LogicGate("NAND_GATE", xdata, NAND_tdata)
NAND_obj.train()

print(NAND_obj.name)
for input_data in test_data:
    (sigmoid_val, logical_val) = NAND_obj.predict(input_data)
    print(input_data, "= ", logical_val, "\n")

# XOR Gate prediction => 예측되지 않는다!
XOR_obj = LogicGate("XOR_GATE", xdata, XOR_tdata)
XOR_obj.train()

print(XOR_obj.name)
for input_data in test_data:
    (sigmoid_val, logical_val) = XOR_obj.predict(input_data)
    print(input_data, " = ", logical_val, "\n")