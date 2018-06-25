import os
import datetime
from copy import deepcopy
import math
import random
def days(date):
    year=int(date[0:4])
    month=int(date[5:7])
    day=int(date[8:10])
    return year, month, day

def be_days(date0, date1):
    days1 = datetime.date(days(date0)[0], days(date0)[1], days(date0)[2])
    days2 = datetime.date(days(date1)[0], days(date1)[1], days(date1)[2])
    if str(days1) == str(days2):
        return 0
    else:
        return int(str(abs(days2-days1)).split(' ')[0])
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
        for j in range(be_days(historyTime_Begin,historyTime_End)):
            trainData[i].append(0)
    for i in ecs_array:
        space_1 = i.find('\t')
        space_2 = i.find('\t', space_1+1)
        tempFlavor = int(i[space_1+7:space_2])
        tempTime = i[space_2+1:].replace('\r\n', '')
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

# def read_lines(file_path):
#   if os.path.exists(file_path):
#       array = []
#       with open(file_path, 'r') as lines:
#           for line in lines:
#               if line !='\r\n':
#                   array.append(line)
#       return array
#   else:
#       print 'file not exist: ' + file_path
#       return None
# escDataPath = '/Users/caozhongjian/Desktop/trainData.txt'
# inputFilePath = '/Users/caozhongjian/Desktop/input.txt'
# resultFilePath = '/Users/caozhongjian/Desktop/output.txt'
#
# esc_infor_array = read_lines(escDataPath)
# input_file_array = read_lines(inputFilePath)

def predict_vm(ecs_lines, input_lines):
    result = []

    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print 'input file information is none'
        return result

    trainData , pcpu, pmem, predict_begintime, predict_endtime, flavorNum, Dimop, flavorL, flavorList= readData(ecs_lines,input_lines)
    # print trainData
    # print shapeSize(trainData)
    finalData = deepcopy(trainData)
    N = be_days(predict_begintime,predict_endtime)
    N1 = len(trainData[0])/N
    N2 = len(trainData[0])%N
    x = [[] for _ in range(flavorNum)]
    y = [[] for _ in range(flavorNum)]
    flavorDs = []
    for i in finalData:
        t = []
        for j in range(N1):
            tt = sum(i[N2+(j*N):(j+1)*N+N2])
            t.append(tt)
        flavorDs.append(t)
    print flavorDs
    flavorDd = []
    beta = random.uniform(0.4,0.95)
    for row in flavorDs:
        mm = int(round(random.uniform(2,N1)))
        v = sum(row[0:mm])/(mm * 1.0)
        t = []
        for j in range(len(row)):
            v = beta*row[j] + (1 - beta) * v
            t.append(v)
        flavorDd.append(t)
    flavorD = []
    for i in flavorDd:
        flavorD.append(round(i[-1]))
    print flavorD


    # x[i].append(finalData[i][j:j+N])
    # y[i].append(finalData[i][j+N])
    # normlizeData = []
    # Meana = []
    # for i in finalData:
    #     n = int(len(i))
    #     dataMean = reduce(lambda x, y: x + y, i) / n
    #     Meana.append(dataMean)
    #     print round(dataMean)
    #     dev = [(math.pow(j - dataMean, 2)) for j in i]
    #     dev = reduce(lambda x, y: x + y, dev) / n
    #     t = []
    #     for ii in i:
    #         tt = abs(ii - dataMean)
    #         t.append(tt)
    #     normlizeData.append(t)

    # for i in range(flavorNum):
    #     for j in range(N, len(normlizeData[0]) - N):
    #         x[i].append(normlizeData[i][j - N:j])
    #         y[i].append(normlizeData[i][j + N])
    # x_test = []
    # for i in normlizeData:
    #     t = []
    #     t.append(i[-N:])
    #     x_test.append(t)
    # yy = [[] for _ in range(flavorNum)]
    # for i in range(flavorNum):
    #     for k in y[i]:
    #         yy[i].append([k])

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
    CflavorD = []
    for i in range(len(flavorList)):
        Acpu.append(stand[i][0])
        Bram.append(stand[i][1])
        CflavorD.append(int(flavorD[i]))

    Vir_Request, N_host, cpurate, ramrate, cpueffvir, rameffvir = assembly_result(Acpu, Bram, CflavorD,Dimop, pcpu, pmem)
    # output.txt
    strline = str(C) + '\r\n'
    for i in range(len(flavorList)):
        strline += 'flavor' + str(flavorList[i]) + ' ' + str(int(flavorD[i])) + '\r\n'

    # strline += 'predict data\n'
    strline += '\r\n' + str(N_host) + '\r\n'

    if Dimop == 'CPU' or Dimop == 'CPU\n':
        for i, x in enumerate(cpueffvir):
            strline += str(i + 1) + ' '
            for j, y in enumerate(x):
                if y != 0:
                    strline += 'flavor' + str(flavorList[j]) + ' ' + str(y) + ' '
            strline += '\r\n'
    elif Dimop == 'MEM' or Dimop == 'MEM\n':
        for i, x in enumerate(rameffvir):
            strline += str(i + 1) + ' '
            for j, y in enumerate(x):
                if y != 0:
                    strline += 'flavor' + str(flavorList[j]) + ' ' + str(y) + ' '
            strline += '\r\n'
    return strline.split('\r\n')
    #return strline
##################fundtion define
##################fundtion define

def resborder(N_host, pCPU, pMEM):
    return int(N_host * pCPU * pMEM)


def min_N_host(vir_request, pCPU, pMEM):
    return int(math.ceil(min(vir_request / pCPU, vir_request / pMEM)))


def max_N_host(vir_request, pCPU, pMEM):
    return int(math.ceil(max(vir_request / pCPU, vir_request / pMEM)))


def multlists(list1, list2):
    if len(list1) == len(list2) and list1!=[] and list2!=[] and type(list1[0]) != list and type(list2[0]) != list :
        productlist = []
        for i in range(len(list1)):
            productlist.append(list1[i] * list2[i])
        return productlist
    else:
        return [0,0]




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
                if j >= k * w[i - 1] and optp[i][j] < optp[i - 1][j - k * w[i - 1]] + k * v[i - 1] and optp[i - 1][j - k * w[i - 1]] + k * v[i - 1] < m1:
                    optp[i][j] = optp[i - 1][j - k * w[i - 1]] + k * v[i - 1]  #
                    if i >= 1 and j >= 1:
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
    # cntr = 0
    flag = True
    while flag:
        for i in range(len(pkcount) - 1,0,- 1):
            for j in range(len(pkcount[i]) - 1 - bias, 0, - 1):
                if optp[i][j] !=optp[i - 1][j] and optp[i][j] == optp[i - 1][j - pkcount[i][j] * w[i - 1]] + pkcount[i][j] * v[i - 1] and pkcount[i][j] != 0:
                    result[i - 1] = pkcount[i][j]
                    if maxvalue - result[i-1] * v[i-1] >= 0:
                        maxvalue -= result[i-1] * v[i-1]
                        bias = len(pkcount[i]) - j + pkcount[i][j] * w[i - 1]
                        print bias
                        break
                    else:
                        result[i-1] = 0
        if sum(multlists(result,w))<= pres and sum(multlists(result,v)) <= max_value:
            flag = False
                    # return result
            break
        else:
            print max_value
            print sum(multlists(result,w))
            print sum(multlists(result,v))
            print 'retry'
            # cntr += 1
    return result


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
