import random
import math
import time
import datetime
import os
from copy import deepcopy


def shapeSize(lists):
    tup = ()

    def Shape_list(lists):
        shapesize = [(len(lists))]
        for i in lists:
            if type(i) == list:
                # shapesize.append(len(lists))
                shapesize.append(Shape_list(i))
            # else:
            # shapesize.append(len(lists))
            break
        return shapesize

    def Shape_tupl(lists):
        for i in lists:
            tmp = (i,)
            if len(lists) != 1:
                tmp = tmp + Shape_tupl(lists[1])
            break
            # else:
            #    tmp=(i[1])
        return tmp

    tup = tup + Shape_tupl(Shape_list(lists))
    return tup


def sub(n, A):
    if type(n) not in (int, float, complex):
        print 'data type error'
        return
    if type(A) not in (list, tuple, range):
        print 'data type error'
        return
    a = []
    for i in A:
        t = []
        for j in i:
            tt = n - j
            t.append(tt)
        a.append(t)

    return a


'''
a = sub(1,[[1,2,3],[4,5,6],[7,8,9]])
print a
'''


def tanh(x):
    if type(x) not in (list, tuple, range):
        print 'type data error'
        return
    tx = []
    cache = x
    for row in x:
        t = []
        for j in row:
            tt = math.tanh(j)
            t.append(tt)
        tx.append(t)
    return tx, cache


def tanh_back(dA, cache):
    Z = cache
    y, caca = tanh(Z)
    y1 = sub(1, y)
    y2 = []
    for i in y:
        t = []
        for j in i:
            tt = j + 1
            t.append(tt)
        y2.append(t)
    dZ = []
    for i in range(len(dA)):
        t = []
        da = dA[i]
        dy2 = y2[i]
        dy1 = y1[i]
        for j in range(len(da)):
            tt = da[j] * dy2[j] * dy1[j]
            t.append(tt)
        dZ.append(t)
    assert (shapeSize(dZ) == shapeSize(Z))
    return dZ


def sigmoid(x):
    ax = []
    cache = x
    if type(x) in (list, tuple, range):
        xt = list(map(list, zip(*x)))
        for row in x:
            t = []
            for j in row:
                tt = 1 / (1 + math.exp(-j))
                t.append(tt)
            ax.append(t)
        return ax, cache
    elif type(x) in (int, float, complex):
        return 1 / (1 + math.exp(-x)), cache
    else:
        print 'data type error'


def sigmoid_back(dA, cache):
    Z = cache
    s, caca = sigmoid(Z)
    s1 = sub(1, s)
    dZ = []
    for i in range(len(dA)):
        da = dA[i]
        ds = s[i]
        ds1 = s1[i]
        t = []
        for j in range(len(da)):
            tt = da[j] * ds[j] * ds1[j]
            t.append(tt)
        dZ.append(t)

    assert (shapeSize(dZ) == shapeSize(Z))
    return dZ


'''
print sigmoid_back([[1,2,3],[4,5,6],[7,8,9]],[[1,3,3],[4,7,6],[7,4,9]])
'''


def relu(x):
    A = []
    for i in x:
        t = []
        for j in i:
            tt = max(0, j)
            t.append(tt)
        A.append(t)
    cache = x
    return A, cache


def relu_back(dA, cache):
    Z = cache
    dZ1 = dA
    dZ = []
    for i in dZ1:
        t = []
        for j in i:
            tt = max(0, j)
            t.append(tt)
        dZ.append(t)
    assert (shapeSize(dZ) == shapeSize(Z))

    return dZ


def matrixReshape(M, size):
    if type(M) not in (list, tuple, range) or type(size) not in (list, tuple, range):
        print 'data type error'
        return
    b = [t for i in M for t in i]
    rM = []
    for i in range(size[0]):
        t = []
        for j in range(size[1]):
            t.append(b[size[1] * i + j])
        rM.append(t)
    return rM


'''
a = [[1,2],[3,4],[3,5]]
data = [[3,4,6],[4,5,6]]
print matrixReshape(a, shapeSize(data))
'''


def matrixDivide(x, y):
    if type(x) not in (list, tuple, range) or type(y) not in (list, tuple, range):
        print 'data type error'
        return
    assert (shapeSize(x) == shapeSize(y))
    div = []
    for i in range(shapeSize(x)[0]):
        t = []
        for j in range(shapeSize(x)[1]):
            try:
                tt = (x[i][j] * 1.0) / y[i][j]
            except:
                tt = (x[i][j] * 1.0) / 0.0001
            t.append(tt)
        div.append(t)
    return div


'''
print matrixDivide([[3,4,5]],[[5,6,7]])
'''


def wmatrix(x, y):
    if type(x) not in (int, float, complex) or type(y) not in (int, float, complex):
        print 'data type error'
        return
    w = [[] for i in range(x)]
    for i in range(x):
        for j in range(y):
            w[i].append(random.random()/x)
            # print w[i]
    return w


def bmatrix(bx):
    if type(bx) not in (int, float, complex):
        print 'data type error'
        return
    b = []
    for i in range(bx):
        b.append([0.001])
    return b


# matrix multityply
def dot(w, a):
    z = []
    at = list(map(list, zip(*a)))
    for row in w:
        t = []
        for col in at:
            tt = sum(i * j for i, j in zip(row, col))
            t.append(tt)
        z.append(t)
    return z


# '''
# print dot([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2,5,7], [1,1,1], [3,3,4]])
# '''


def addMatrix(x, y):
    if type(x) not in (list, tuple, range) or type(y) not in (list, tuple, range):
        print 'data type error!!!!'
        return
    assert (shapeSize(x) == shapeSize(y))
    addM = []
    for i in range(shapeSize(x)[0]):
        t = []
        for j in range(shapeSize(x)[1]):
            tt = x[i][j] + y[i][j]
            t.append(tt)
        addM.append(t)
    return addM

def subMatrix(x, y):
    if type(x) not in (list, tuple, range) or type(y) not in (list, tuple, range):
        print 'data type error!!!!'
        return
    assert (shapeSize(x) == shapeSize(y))
    addM = []
    for i in range(shapeSize(x)[0]):
        t = []
        for j in range(shapeSize(x)[1]):
            tt = x[i][j] - y[i][j]
            t.append(tt)
        addM.append(t)
    return addM


# '''
# print addMatrix([[1,2,3],[2,3,4]],[[6,5,6],[7,8,9]])
# '''


def log(A):
    a = []
    if type(A) not in (list, tuple, range):
        print 'data type error'
    else:
        for i in A:
            t = []
            for j in i:
                try:
                    tt = math.log(j)
                except:
                    tt = math.log(0.001)
                t.append(tt)
            a.append(t)
    return a


# '''
# a = log([[1,2,3],[4,5,6],[7,8,9]])
# print a
# '''


def initialize_parameters_deep(layers):
    parameters = {}
    L = len(layers)

    for i in range(1, L):
        parameters['W' + str(i)] = wmatrix(layers[i], layers[i - 1])
        parameters['b' + str(i)] = bmatrix(layers[i])
        assert (shapeSize(parameters['W' + str(i)]) == (layers[i], layers[i - 1]))
        assert (shapeSize(parameters['b' + str(i)]) == (layers[i], 1))

    return parameters


# pra = initialize_parameters_deep([2, 7, 21, 4])

def linear_forward(A, W, b):
    Z1 = dot(W, A)
    Z = []
    # transposing
    #    bT = list(map(list, zip(*b)))
    bT = [t for i in b for t in i]
    ZT = list(map(list, zip(*Z1)))
    # print ZT
    for i in range(len(Z1)):
        t = []
        for ii in range(len(ZT)):
            tt = Z1[i][ii] + bT[i]
            t.append(tt)
        Z.append(t)
    # print shapeSize(Z)
    assert (shapeSize(Z) == (shapeSize(W)[0], shapeSize(A)[1]))
    cache = (A, W, b)

    return Z, cache


# '''
# z,b = linear_forward([[1], [2], [3]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2], [1], [3]])
# print z
# '''


def activation_forward(A_prev, W, b, activation):
    if activation == 'tanh':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    assert (shapeSize(A) == (shapeSize(W)[0], shapeSize(A_prev)[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# '''
# a, b = activation_forward([[1] , [2], [3]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2], [1], [3]], 'sigmoid')
# print a
# print b
# '''


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L + 1):
        A_prev = A
        A, cache = activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    ALR = []  # change the float value into type int!!!
    for row in A:
        t = []
        for i in row:
            tt = round(i)
            t.append(tt)
        ALR.append(t)

    # print shapeSize(AL)

    # assert (shapeSize(AL) == (shapeSize(X)[0], 1))

    return ALR, caches


# '''
# aa,bb = L_model_forward([[1],[2]],pra)
# att = list(map(list, zip(*aa)))
# print att
# '''


def compute_cost(AL, Y):
    m = shapeSize(Y)[1]
    s1 = dot(log(AL), Y)
    s2 = dot(sub(1, log(AL)), sub(1, Y))
    ss = 0
    for i in range(len(AL)):
        s = reduce(lambda x, y: x + y, s1[i]) + reduce(lambda x, y: x + y, s2[i])
        ss += s
    cost = -ss / m
    return cost

def square_cost(AL, Y):
    m = shapeSize(Y)[1]
    Yt = matrixReshape(Y, shapeSize(AL))
    t = subMatrix(AL, Yt)

    sum = 0
    for i in range(shapeSize(t)[0]):
        for j in range(shapeSize(t)[1]):
            tt = math.pow(t[i][j], 2)
            sum  = sum + 0.5*tt
    cost = sum/m
    return cost


def mul(A, n):
    a = []
    if type(A) not in (list, tuple, range):
        print 'data error'
        return
    if type(n) not in (int, float, complex):
        print 'data error'
        return
    for i in A:
        t = []
        for j in i:
            tt = j * n
            t.append(tt)
        a.append(t)
    return a


# '''
# print mul([[1,2,3], [4,5,6], [7,8,9]],1.0/9)
# '''
# '''
# print compute_cost([[1,2,3], [4,5,6], [7,8,9]], [[1],[2],[3]])
# '''


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = shapeSize(A_prev)[1]
    A_prevT = list(map(list, zip(*A_prev)))
    dW = mul(dot(dZ, A_prevT), 1.0 / m)
    dbb = []
    for i in dZ:
        t = []
        tt = reduce(lambda x, y: x + y, i)
        t.append(tt)
        dbb.append(t)
    db = mul(dbb, 1.0 / m)
    WT = list(map(list, zip(*W)))
    dA_prev = dot(WT, dZ)
    assert (shapeSize(dA_prev) == shapeSize(A_prev))
    assert (shapeSize(dW) == shapeSize(W))
    assert (shapeSize(db) == shapeSize(b))

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_back(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'tanh':
        dZ = tanh_back(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# '''
# print linear_activation_backward([[1,2,3],[4,5,6],[7,8,9]],(([[1,3,3],[2,3,4],[3,2,1]],[[4,3,4],[5,4,5],[6,5,6]],[[3],[5],[7]]), [[1,3,3],[4,7,6],[7,4,9]]),'sigmoid')
# '''


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = shapeSize(AL)[1]
    Y1 = matrixReshape(Y, shapeSize(AL))
    # print mul(matrixDivide(Y1, AL), -1)
    # print mul(matrixDivide(sub(1,Y1),sub(1,AL)), -1)
    dAL = addMatrix(mul(matrixDivide(Y1, AL), -1), matrixDivide(sub(1, Y1), sub(1, AL)))
    #dAL = subMatrix(AL, Y1)
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  "relu")
    for l in reversed(range(L - 1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network
    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = addMatrix(parameters["W" + str(l + 1)],
                                                 mul(grads["dW" + str(l + 1)], -learning_rate))
        parameters["b" + str(l + 1)] = addMatrix(parameters["b" + str(l + 1)],
                                                 mul(grads["db" + str(l + 1)], -learning_rate))

    return parameters


def predict(X, y, parameters):
    m = shapeSize(X)
    n = len(parameters) // 2
    # pt = bmatrix(m)
    # p = list(map(list, zip(*pt)))
    probas, caches = L_model_forward(X, parameters)
    count = []
    acc = 0
    for i in probas:
        for j in range(len(i)):
            if i[j] == y[j]:
                count.append(1)

        #print "Accuracy:" + str((len(count) * 1.0) / len(i))
        acc = (len(count) * 1.0) / len(i)
    return probas, acc


def predict_x(X, parameters):
    probas, caches = L_model_forward(X, parameters)

    return probas


time1 = time.time()


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations):
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        #print square_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

    return parameters


def days(date):
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    return year, month, day


def be_days(date0, date1):
    days1 = datetime.date(days(date0)[0], days(date0)[1], days(date0)[2])
    days2 = datetime.date(days(date1)[0], days(date1)[1], days(date1)[2])
    if str(days1) == str(days2):
        return 0
    else:
        return int(str(abs(days2 - days1)).split(' ')[0])


def readData(ecs_array, input_array):
    pline1 = input_array[0]
    pspace_1 = pline1.find(' ')
    pspace_2 = pline1.find(' ', pspace_1 + 1)
    pCPU = int(pline1[0:pspace_1])
    pMEM = int(pline1[pspace_1:pspace_2])
    FlavorNum = int(input_array[1])
    flavorL = input_array[2:2 + FlavorNum]
    flavorList = []
    for i in flavorL:
        Space_1 = i.find(' ')
        Space_2 = i.find(' ', Space_1 + 1)
        Space_3 = i.find('\r\n', Space_2 + 1)
        NUM = int(i[6:Space_1])
        #        CPU = int(i[Space_1:Space_2])
        #        MEM = int(i[Space_2:Space_3])
        flavorList.append(NUM)
    flavorList = sorted(flavorList)
    #print flavorList
    DimToBeOptimized = input_array[-3].replace('\r\n', '')
    PredictTime_Begin = input_array[-2].replace('\r\n', '')
    PredictTime_End = input_array[-1].replace('\r\n', '')
    bline = ecs_array[0]
    Space_1 = bline.find('\t')
    Space_2 = bline.find('\t', Space_1 + 1)
    historyTime_Begin = bline[Space_2 + 1:].replace('\r\n', '')

    fline = ecs_array[-1]
    Space_3 = fline.find('\t')
    Space_4 = fline.find('\t', Space_3 + 1)
    historyTime_End = fline[Space_4 + 1:].replace('\r\n', '')

    trainData = [[0] for _ in range(FlavorNum)]
    for i in range(FlavorNum):
        for j in range(be_days(historyTime_Begin, historyTime_End)):
            trainData[i].append(0)
    for i in ecs_array:
        space_1 = i.find('\t')
        space_2 = i.find('\t', space_1 + 1)
        tempFlavor = int(i[space_1 + 7:space_2])
        tempTime = i[space_2 + 1:].replace('\r\n', '')
        if tempTime is not None:
            value = be_days(historyTime_Begin, tempTime)
            if tempFlavor in flavorList:
                ii = flavorList.index(tempFlavor)
                trainData[ii][value] += 1
            else:
                None
        else:
            print('Time data error.\n')

    return trainData, pCPU, pMEM, PredictTime_Begin, PredictTime_End, FlavorNum, DimToBeOptimized, flavorL, flavorList


##############################################################
# def read_lines(file_path):
#    if os.path.exists(file_path):
#        array = []
#        with open(file_path, 'r') as lines:
#            for line in lines:
#                if line !='\r\n':
#                    array.append(line)
#        return array
#    else:
#        print 'file not exist: ' + file_path
#        return None
# escDataPath = '/home/qun/workspace/TrainData.txt'
# inputFilePath = '/home/qun/workspace/input.txt'
# resultFilePath = '/home/qun/workspace/output.txt'
# 
# esc_infor_array = read_lines(escDataPath)
# input_file_array = read_lines(inputFilePath)
#############################################################

def predict_vm(ecs_lines, input_lines):
    result = []

    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print 'input file information is none'
        return result

    trainData, pcpu, pmem, predict_begintime, predict_endtime, flavorNum, Dimop, flavorL, flavorList = readData(
        ecs_lines, input_lines)
    # print trainData
    # print shapeSize(trainData)
    finalData = deepcopy(trainData)
    N = be_days(predict_begintime, predict_endtime)
    x = [[] for _ in range(flavorNum)]
    y = [[] for _ in range(flavorNum)]
    ##########################data process

    for i in range(flavorNum):
        for j in range(N, len(trainData[0])):
            finalData[i][j] = sum(trainData[i][j - N:j])
            # x[i].append(finalData[i][j:j+N])
            # y[i].append(finalData[i][j+N])
    normlizeData = []
    Meana = []
    for i in finalData:
        n = int(len(i))
        dataMean = reduce(lambda x, y: x + y, i) / n
        Meana.append(dataMean)
        #print round(dataMean)
        dev = [(math.pow(j - dataMean, 2)) for j in i]
        dev = reduce(lambda x, y: x + y, dev) / n
        t = []
        for ii in i:
            tt = abs(ii - dataMean)
            t.append(tt)
        normlizeData.append(t)

    for i in range(flavorNum):
        for j in range(N, len(finalData[0]) - N):
            x[i].append(finalData[i][j:j + N])
            y[i].append(finalData[i][j + N])
    x_test = []
    for i in finalData:
        t = []
        t.append(i[-N:])
        x_test.append(t)
    yy = [[] for _ in range(flavorNum)]
    for i in range(flavorNum):
        for k in y[i]:
            yy[i].append([k])
    flavorD = []
    for i in range(flavorNum):
        xx = list(map(list, zip(*x[i])))
        xx_test = list(map(list, zip(*x_test[i])))
        prameters = L_layer_model(xx, yy[i], [N, 24, 12, 1], 0.00001, 500)
        yp = predict_x(xx_test, prameters)
        yp = [t for i in yp for t in i]
        flavorD.append(yp[0])
    flavorD = [(x + y) for x, y in zip(flavorD, Meana)]
    print flavorD
    stand = []
    for i in flavorL:
        Space_1 = i.find(' ')
        Space_2 = i.find(' ', Space_1 + 1)
        Space_3 = i.find('\r\n', Space_2 + 1)
        # NUM = int(i[6:Space_1])
        CPU = int(i[Space_1:Space_2])
        MEM = int(i[Space_2:Space_3]) / 1024
        stand.append([CPU, MEM])
    stand = sorted(stand)
    # Vir_Request = [sum(multlists(C, A)), sum(multlists(C, B))]
    # main()

    Acpu = []
    Bram = []
    # C = [sum(i) for i in historyData]
    C = int(sum(flavorD))
    for i in range(len(flavorList)):
        flavorD[i] = int(flavorD[i])
        Acpu.append(stand[i][0])
        Bram.append(stand[i][1])

    strline = str(C) + '\r\n'
    for i in range(len(flavorList)):
        strline += 'flavor' + str(flavorList[i]) + ' ' + str(int(flavorD[i])) + '\r\n'
    hosts = hosts_allocate(flavorD, flavorList, stand, Dimop, pcpu, pmem)
    strline += '\r\n' + str(len(hosts)) + '\r\n'
    tid = 0
    for i in hosts:
        tid += 1
        strline += str(tid) + ' '
        for j in range(len(i)):
            strline += 'flavor' + str(i[j][0]) + ' ' + str(i[j][1]) + ' '
        strline += '\r\n'

    Vir_Request, N_host, cpurate, ramrate, cpueffvir, rameffvir = assembly_result(Acpu, Bram, flavorD, pcpu, pmem,Dimop)
       # output.txt
    strline = str(C) + '\r\n'
    for i in range(len(flavorList)):
        strline += 'flavor' + str(flavorList[i]) + ' ' + str(int(flavorD[i])) + '\r\n'

       # strline += 'predict data\n'
    strline += '\r\n' + str(N_host) + '\r\n'

    if Dimop == 'CPU':
        for i, x in enumerate(cpueffvir):
            strline += str(i + 1) + ' '
            for j, y in enumerate(x):
                if y != 0:
                    strline += 'flavor' + str(flavorList[j]) + ' ' + str(y) + ' '
            strline += '\r\n'
    elif Dimop == 'MEM':
        for i, x in enumerate(rameffvir):
            strline += str(i + 1) + ' '
            for j, y in enumerate(x):
                if y != 0:
                    strline += 'flavor' + str(flavorList[j]) + ' ' + str(y) + ' '
            strline += '\r\n'
    return strline.split('\r\n')


##################fundtion define
def sort(stand, flavorList, dim2op):
    stand = sorted(stand, reverse=True)
    flavorList = sorted(flavorList, reverse=True)
    if dim2op == 'CPU':
        for mm in range(len(stand)):
            for i in range(len(stand) - 1):
                if stand[i][0] == stand[i + 1][0]:
                    if stand[i][1] > stand[i + 1][1]:
                        stand[i], stand[i + 1] = stand[i + 1], stand[i]
                        flavorList[i], flavorList[i + 1] = flavorList[i + 1], flavorList[i]
    if dim2op == 'MEM':
        for i in range(len(stand)):
            for j in range(i + 1, len(stand)):
                if stand[i][1] < stand[j][1]:
                    stand[i], stand[j] = stand[j], stand[i]
                    flavorList[i], flavorList[j] = flavorList[j], flavorList[i]
                elif stand[i][1] == stand[j][1]:
                    if stand[i][0] > stand[j][0]:
                        stand[i], stand[j] = stand[j], stand[i]
                        flavorList[i], flavorList[j] = flavorList[j], flavorList[i]
    return stand, flavorList


def hosts_allocate(flavorD, flavorList, stand, dimop, pcpu, pmem):
    reflavorD = []
    restand = []
    reflavorList = []

    if dimop == 'CPU':
        restand, reflavorList = sort(stand, flavorList, dimop)
        for i in reflavorList:
            t = flavorList.index(i)
            reflavorD.append(flavorD[t])
    if dimop == 'MEM':
        restand, reflavorList = sort(stand, flavorList, dimop)
        for i in reflavorList:
            t = flavorList.index(i)
            reflavorD.append(flavorD[t])
    a = []
    for i in range(len(restand)):
        tt = []
        for j in restand[i]:
            tt.append(j)
        tt.insert(0, reflavorD[i])
        a.append(tt)
    b = [pcpu, pmem]
    Flag = 1
    hosts = []
    m = 0
    while (Flag == 1):
        host = []

        if (Flag == 1):
            flag = 0
            for i in range(0, len(a)):
                if (a[i][0] > 0):
                    t = [reflavorList[i], 0]
                    for j in range(0, a[i][0]):
                        if ((b[0] > 0 and b[0] < a[i][1]) or (b[1] > 0 and b[1] < a[i][2])):
                            break
                        b[0] -= a[i][1]
                        b[1] -= a[i][2]
                        a[i][0] -= 1
                        t[1] += 1
                        if (b[0] == 0 or b[1] == 0):
                            flag = 1
                            break
                    if (t[1] > 0):
                        host.append(t)
                    if (flag == 1):
                        b = [pcpu, pmem]
                        flag = 0
                        break
        # print (a)
        hosts.append(host)
        b = [pcpu, pmem]
        for i in range(0, len(a)):
            if (a[i][0] != 0):
                Flag = 1
                break
            else:
                Flag = 0
    return hosts


##################fundtion define

def resborder(N_host, pCPU, pMEM):
    return int(N_host * pCPU * pMEM)


def min_N_host(vir_request, pCPU, pMEM):
    return int(math.ceil(min(vir_request / pCPU, vir_request / pMEM)))


def max_N_host(vir_request, pCPU, pMEM):
    return int(math.ceil(max(vir_request / pCPU, vir_request / pMEM)))


def multlists(list1, list2):
    if len(list1) == len(list2) and list1 != [] and list2 != [] and type(list1[0]) != list and type(list2[0]) != list:
        productlist = []
        for i in range(len(list1)):
            productlist.append(list1[i] * list2[i])
        return productlist
    else:
        return [0, 0]




def MultiplePack(n, m, w, v, num, m1):
    optp = [[0 for col in range(m + 1)] for row in range(n + 1)]
    pkcount = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    pkg = [[ 0 for _ in range(m)]for _ in range(n)]
    virs_count = [0 for row in range(n)]
    for i in range(1, n + 1):
        # print (time.ctime())
        for j in range(1, m + 1):
            # if optp[i][j]<m1:
            max_num_i = int(min(math.floor(1.0 * j / w[i - 1]), num[i - 1]))
            optp[i][j] = optp[i - 1][j]
            # if optp[i][j]<m1:
            for k in range(max_num_i + 1):
                if j > k * w[i - 1] and optp[i][j] < optp[i - 1][j - k * w[i - 1]] + k * v[i - 1] and optp[i - 1][
                            j - k * w[i - 1]] + k * v[i - 1] <= m1:
                    optp[i][j] = optp[i - 1][j - k * w[i - 1]] + k * v[i - 1]  #
                    if i >= 1 and j >= 1  :
                        pkcount[i][j] = k


    max_value = optp[n][m]
    virs_count = exchange(optp,max_value,  pkcount,w, v, m)
    return max_value, virs_count




# def exchange(max_value, weight, num, pres) :
#     result = [0 for _ in range(len(weight))]
#     I,B = index_sort(num, True)
#     for i in range(len(num)) :
#         if max_value / weight[I[i]] <= B[i]:
#             result[I[i]] = max_value / weight[I[i]]
#             max_value = max_value // weight[I[i]]
#             if max_value <= 0:
#                 break
#         elif max_value - weight[I[i]] * B[i] > 0:
#             for k in range(B[i]):
#                 result[I[i]] = B[i]
#                 max_value = max_value - weight[I[i]] * B[i]
#                 if max_value <= 0:
#                     break
#
#     return result


def index_sort ( aim, TF = False):
    b = sorted(aim ,reverse= TF)
    bindex = [0 for _ in range(len(aim))]
    for i, x in enumerate(aim):
        for j,y in enumerate(b) :
            if x == y :
                bindex[j] = i
                b[j]=None
                break
    return bindex ,sorted(aim, reverse= TF)



def exchange(optp, max_value, pkcount, w, v, pres):
    maxvalue= deepcopy(max_value)
    result = [0 for _ in range(len(w))]
    bias = 0
    for i in range(len(pkcount)-1,0,-1):
        for j in range(len(pkcount[i])-1-bias,0,-1):
            if optp[i][j] == optp[i - 1][j - pkcount[i][j] * w[i - 1]] + pkcount[i][j] * v[i - 1] :
                result[i-1] = pkcount[i][j]
                if maxvalue - result[i-1] * v[i-1] >= 0:
                    maxvalue -= result[i-1] * v[i-1]
                    bias = len(pkcount[i]) - j
                    break
                else:
                    result[i-1] = 0
    if sum(multlists(result,w))<= pres and sum(multlists(result,v)) == max_value:
        return result
    else:
        print sum(multlists(result,w))
        print sum(multlists(result,v))
        print 'erro'
        exit(1)



def assembly_result(A, B, C ,dimop,pcpu,pmen):
    Vir_Request = [sum(multlists(C, A)), sum(multlists(C, B))]
    N_host = int(math.ceil(max(multlists(Vir_Request, [1 / float(pcpu), 1 / float(pmen)], ))))
    cpurate = []
    cpueffvir = []
    ramrate = []
    rameffvir = []
    costCPU = 0
    costRAM = 0
    cpuvir=[]
    ramvir=[]
    if N_host == 0:
        print ('Dont need phycal machine')
    else:
        N_host = N_host <<1
        for i in range(1, N_host ):

            # if C == [0 for _ in range(len(C))]:
            if sum(C) == 0:
                break
            else:

                if dimop=='CPU'or dimop=='CPU\n':
                    if cpuvir != []:
                        for k in range(0,len(C)):
                            C[k] = C[k] - cpuvir[k]
                    if sum(C) == 0:
                        break
                    costCPU, cpuvir = MultiplePack(len(C), pmen, B, A, C, pcpu)  # costCPU+56
                    cpueffvir.append(cpuvir)
                    cpurate.append(costCPU / float(pcpu))

                elif dimop == 'MEM' or dimop == 'MEM\n':
                    if ramvir!=[]:
                        for k in range(0,len(C)):
                            C[k] -= ramvir[k]
                    if sum(C) == 0:
                        break
                    costRAM, ramvir = MultiplePack(len(C), pcpu, A, B, C, pmen)  # costRAM+128
                    rameffvir.append(ramvir)
                    ramrate.append(costRAM / float(pmen))

            # print('when host='+str(i))
            # print('server:'+str(i*pCPU)+'and'+str(i*pMEM))
            # print('VM:'+str(costCPU)+'and'+str(costRAM))
            #

            # if costCPU <= i * pCPU:
            #     cpurate.append(costCPU / i / float(pCPU))
            #     cpueffvir.append(cpuvir)
            # if costRAM <= i * pMEM:
            #     ramrate.append(costRAM / i / float(pMEM))
            #     rameffvir.append(ramvir)

    # for i in range(len(cpurate)-1, 0, -1):
    #     for j in range(len(C)):
    #         cpueffvir[i][j] = cpueffvir[i][j] - cpueffvir[i - 1][j]
    #         rameffvir[i][j] = rameffvir[i][j] - rameffvir[i - 1][j]
    # print score

    # score=[]
    # for i in rate:
    #     score.append(f(rate))

    print ('CPU' + 'OR RAM ')
    # for i in range(len(cpurate)):
    #     print (str(cpurate[i]) + '\t\t' + str(ramrate[i]))
    print ('CPU  vm  list:')
    print(cpueffvir)
    print ('RAM  vm  list:')
    print(rameffvir)
    #N_host if host-1

    for i in range(len(cpueffvir)-1,0,-1):
        if cpueffvir[-i] == [0 for col in range(len(cpueffvir[0]))]:
            cpueffvir.pop()
    for i in range(len(rameffvir) - 1, 0, -1):
        if rameffvir[-i] == [0 for col in range(len(rameffvir[0]))]:
            rameffvir.pop()
    N_host=max(len(cpueffvir),len(rameffvir))
    return Vir_Request, N_host, cpurate, ramrate, cpueffvir, rameffvir
# f = open(resultFilePath, 'w+')
# strline = predict_vm(esc_infor_array, input_file_array)
# f.writelines(strline)
# #print predict_vm(esc_infor_array, input_file_array)
# f.close()


