#!/usr/bin/env python
import numpy as np

def test_q1(q1):
    A = np.array([[ 0.447005  ,  0.21022493,  0.93728845,  0.83303825,  0.08056971],
        [ 0.00610984,  0.82094051,  0.97360379,  0.35340124,  0.30015068],
        [ 0.20040093,  0.8435867 ,  0.1604327 ,  0.02812206,  0.79177117],
        [ 0.46312438,  0.21134326,  0.54136079,  0.14242225,  0.83576312],
        [ 0.31526587,  0.64899011,  0.49244916,  0.63672197,  0.44789057]])
    b = np.array([ 0.60448536,  0.04661395,  0.44346333,  0.8076896 ,  0.85597154])
    x = q1(A,b)
    assert np.allclose(np.dot(A,x), b), 'Sorry, that is the wrong function.'
    print('Question 1 Passed!')

def test_q2(q2):
    A = np.array(range(1,26)).reshape((5,5))
    b = np.array([[2,4],[12,14],[22,24]])
    assert np.array_equal(q2(A), b), 'Question 2 Failed!\nInput:\n%r\nCorrect Answer:\n%r\nYour Answer:\n%r' % (A, b, q2(A))
    print('Question 2 Passed!')

def test_q3(q3):
    A = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,1,0]])
    assert q3(A) == 2, 'Question 3 Failed!\nInput:\n%r\nCorrect Answer:\n%r\nYour Answer:\n%r' % (A, 2, q3(A))
    print('Question 3 Passed!')

def test_q4(q4):
    A = np.array([[1,0,0,2],[0,1,0,2],[0,0,1,2]])
    B = np.array([[1,2,3],[4,5,6],[7,8,9]])
    Output = np.array([[1,2,3,1,0],[4,5,6,0,0],[7,8,9,0,1]])
    assert np.array_equal(q4(A,B), Output), 'Question 4 Failed!\nInput:\nA: %r\nB: %r\nCorrect Answer:\n%r\nYour Answer:\n%r' % (A, B, Output, q4(A,B))
    print('Question 4 Passed!')

def test_q5(q5):
    a = np.array([2017,4036,6057,8080,10105,12132,14161,16192,18225,20260,22297,24336,26377,28420,30465,32512,34561,36612,38665,40720])
    assert np.array_equal(q5(20), a), 'Question 5 Failed!\nInput: 20\nCorrect Answer:\n%r\nYour Answer:\n%r' % (a, q5(20))
    print('Question 5 Passed!')

def test_q6(q6):
    v = np.array([0,1,2,3,4,5])
    N = 3
    Output = np.array([3,4,5,0,1,2])
    assert np.array_equal(q6(v,N), Output), 'Question 6 Failed!\nInput:\nv: %r\nN: %r\nCorrect Answer:\n%r\nYour Answer:\n%r' % (v, N, Output, q6(v,N))
    print('Question 6 Passed!')

def test_q7(q7):
    assert np.array_equal(q7(np.eye(400),137), np.eye(263)), 'Question 7 Failed!\nInput:\nI(400)\nN: 137\nCorrect Answer:\n%r\nYour Answer:\n%r' % (np.eye(263), q7(np.eye(400),137))
    print('Question 7 Passed!')

def test_q8(q8):
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    Output = [9,5,1]
    assert np.array_equal(q8(A), Output), 'Question 8 Failed!\nInput:\n%r\nCorrect Answer:\n%r\nYour Answer:\n%r' % (A, Output, q8(A))
    print('Question 8 Passed!')


def test_q9(q9):
    A = np.array([[1,2,3],[4,5,6]])
    B = np.array([[1,1],[1,1],[1,1]])
    Output = np.array([[1,2,3],[4,5,6],[0,0,0]]), np.array([[1,1,0],[1,1,0],[1,1,0]])
    assert np.array_equal(q9(A,B), Output), 'Question 9 Failed!\nInput:\nA: %r\nB: %r\nCorrect Answer:\n%r\nYour Answer:\n%r' % (A, B, Output, q9(A,B))
    print('Question 9 Passed!')

def test_q10(q10):
    A = np.array([[1,1,1],[1,1,1],[1,1,1] ,[1,1,1]])
    B = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    #p = 2
    p = 1
    Output = np.array([[1, 1, 1, 4, 8, 12]])
    #Output = np.array([[1,1,1,3,7,11],[1,1,1,4,8,12]])
    assert np.array_equal(q10(A,B,p), Output), 'Question 10 Failed!\nInput:\nA: %r\nB: %r\np: 2\nCorrect Answer:\n%r\nYour Answer:\n%r' % (A, B, Output, q10(A,B,p))
    print('Question 10 Passed!')


def test_all(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10):
    test_q1(q1)
    test_q2(q2)
    test_q3(q3)
    test_q4(q4)
    test_q5(q5)
    test_q6(q6)
    test_q7(q7)
    test_q8(q8)
    test_q9(q9)
    test_q10(q10)

if __name__ == '__main__':
    print('Autograder loaded!\nQuestion 0 Passed!')