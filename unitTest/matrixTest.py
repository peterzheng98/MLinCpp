import numpy as np
import random
import subprocess
'''
Usage: 
The unitTest will put matrix.in and run the program to get the output and compare them with the result of numpy.
Notice that the input is placed in configure.in and matrix.in, the output should be placed in result.out
The arithtest request for all the result which are +, -, *(matrixA and matrixB[0][0]), *(matrixA[0][0] and matrixB)
'''
NumOfNormalArith = 1000 # rounds for normal calculation
LowerBound = -100
UpperBound = 100
runningTimeout = 1
MatrixCppPath = './MatrixTest'
CurrentProgress = 1
REDColor = '\033[31m'
GreenColor = '\033[32m'
BlueColor = '\033[34m'
EndColor = '\033[0m'
PREFIX = 'dump/'
np.set_printoptions(threshold=NumOfNormalArith * NumOfNormalArith + 10, linewidth=NumOfNormalArith + 10)
def dumpOut(msg, result, idx):
    # np.set_printoptions(threshold=NumOfNormalArith * NumOfNormalArith + 10, linewidth=NumOfNormalArith + 10)
    with open(PREFIX+'dump_%d.dmp' % idx, 'w') as f:
        f.write('==========Error Message==========\n%s\n============================\n' % msg)
        for i in result:
            for line in i:
                for col in line:
                    f.write('%d ' % col)
                f.write('\n')
            f.write('\n=================\n')
            




# Test for + between to matrix
for _ in range(NumOfNormalArith):
    print('[' + BlueColor + 'PE' + EndColor + '] Test %d / %d' %(_ + 1, NumOfNormalArith), end='\r')
    matrixA = []
    matrixB = []
    sizeM = random.randint(1, 10)
    sizeN = random.randint(1, 10)
    for __ in range(sizeM):
        matrixA_1 = []
        matrixB_1 = []
        for ___ in range(sizeN):
            matrixA_1.append(random.randint(LowerBound, UpperBound))
            matrixB_1.append(random.randint(LowerBound, UpperBound))
        matrixA.append(matrixA_1)
        matrixB.append(matrixB_1)
    with open('configure.in', 'w') as f:
        f.write('%d %d 0 0' % (sizeM, sizeN)) # 0 0 indicates two matrix are int, int

    with open('matrix.in', 'w') as f:
        for i in matrixA:
            for j in i:
                f.write('%d ' % j)
            f.write('\n')
        f.write('\n')
        for i in matrixB:
            for j in i:
                f.write('%d ' % j)
            f.write('\n')
        f.write('\n')
    
    matrixA_numpy = np.array(matrixA)
    matrixB_numpy = np.array(matrixB)
    result_plus = matrixA_numpy + matrixB_numpy
    result_minus = matrixA_numpy - matrixB_numpy
    result_multi_singleB = np.zeros(matrixA_numpy.shape)
    lineidx = 0
    colidx = 0
    for line in matrixA_numpy:
        for col in line:
            result_multi_singleB[lineidx][colidx] = col * matrixB_numpy[0][0]
            colidx += 1
        colidx = 0
        lineidx += 1
    result_multi_singleA = np.zeros(matrixA_numpy.shape)
    lineidx = 0
    colidx = 0
    for line in matrixB_numpy:
        for col in line:
            result_multi_singleA[lineidx][colidx] = col * matrixA_numpy[0][0]
            colidx += 1
        colidx = 0
        lineidx += 1
    correct = 0
    # Execute the program
    try:
        subprocess.run(MatrixCppPath, timeout = 2)
    except subprocess.TimeoutExpired:
        correct = 3
        dumpOut('Time Limit Exceeded. Dumpout Seq: A, B, A+B, A-B, A*B[0][0], A[0][0]*B', [matrixA_numpy, matrixB_numpy, result_plus, result_minus, result_multi_singleA, result_multi_singleB], _ + 1)
    except Exception as identifier:
        correct = 2
        dumpOut('Runtime Error[1-%s]. Dumpout Seq: A, B, A+B, A-B, A*B[0][0], A[0][0]*B' % identifier, [matrixA_numpy, matrixB_numpy, result_plus, result_minus, result_multi_singleA, result_multi_singleB], _ + 1)
    # Get output here
    if correct == 0:
        try:
            rawData = []
            with open('result.out', 'r') as f:
                _rawData = f.readlines()
                for i in range(len(_rawData)):
                    rawData.append(_rawData[i].strip('\n'))
            if False:
                correct = 1
                dumpOut('Wrong Answer. A Dumpout Seq: A, B, A+B, A-B, A*B[0][0], A[0][0]*B', [matrixA_numpy, matrixB_numpy, result_plus, result_minus, result_multi_singleA, result_multi_singleB], _ + 1)
                pass
            else:
                curIdx = 0
                for round in range(1, 5):
                    if correct == 1:
                        continue
                    thisMatrix = rawData[curIdx:curIdx + sizeM]
                    numpy_matrix_raw = [i.split(' ') for i in thisMatrix]
                    numpy_matrix = []
                    for line in numpy_matrix_raw:
                        numpy_matrix.append([int(i) for i in line[:-1]])
                    numpy_matrix = np.array(numpy_matrix)
                    if round == 1:
                        if np.array_equal(numpy_matrix, result_plus):
                            pass
                        else:
                            correct = 1
                            dumpOut('Wrong Answer. Dumpout Seq: A, B, A+B, your answer', [matrixA_numpy, matrixB_numpy, result_plus, numpy_matrix], _ + 1)
                            pass
                    elif round == 2 and correct == 0:
                        if np.array_equal(numpy_matrix, result_minus):
                            pass
                        else:
                            correct = 1
                            dumpOut('Wrong Answer. Dumpout Seq: A, B, A-B, your answer', [matrixA_numpy, matrixB_numpy, result_minus, numpy_matrix], _ + 1)
                            pass
                    elif round == 3 and correct == 0:
                        if np.array_equal(numpy_matrix, result_multi_singleB):
                            pass
                        else:
                            correct = 1
                            dumpOut('Wrong Answer. Dumpout Seq: A, B, A*B[0][0], your answer', [matrixA_numpy, matrixB_numpy, result_multi_singleA, numpy_matrix], _ + 1)
                            pass
                    elif round == 4 and correct == 0:
                        if np.array_equal(numpy_matrix, result_multi_singleA):
                            pass
                        else:
                            correct = 1
                            dumpOut('Wrong Answer. Dumpout Seq: A, B, A[0][0]*B, your answer', [matrixA_numpy, matrixB_numpy, result_multi_singleB, numpy_matrix], _ + 1)
                            pass
                    curIdx += (sizeM + 1)
        except Exception as identifier:
            correct = 2
            dumpOut('Runtime Error[2-%s]. Dumpout Seq: A, B, A+B, A-B, A*B[0][0], A[0][0]*B' % identifier, [matrixA_numpy, matrixB_numpy, result_plus, result_minus, result_multi_singleA, result_multi_singleB], _ + 1)


    if correct == 0:
        print('[' + GreenColor + 'AC' + EndColor + '] Test %d Accepted' %(_ + 1), end='\n')
    elif correct == 1:
        print('[' + REDColor + 'WA' + EndColor + '] Test %d Wrong Answer, result dumped.' %(_ + 1), end='\n')
    elif correct == 2:
        print('[' + REDColor + 'RE' + EndColor + '] Test %d Runtime Error, result dumped.' %(_ + 1), end='\n')
    elif correct == 3:
        print('[' + REDColor + 'TE' + EndColor + '] Test %d Time limit exceeded, result dumped.' %(_ + 1), end='\n')
