import numpy as np

def testing(H, H_Alt):
  
    htest = ''
    halttest = ''
    H_Correct = np.eye(32*32)
    H_Alt_Correct = np.vstack((H_Correct[::2], H_Correct[1::2]))
    if not np.isfinite(np.linalg.cond(H)) or not np.array_equal(H, H_Correct):
        htest = 'H shape is incorrect'
    if not np.isfinite(np.linalg.cond(H_Alt)) or not np.array_equal(H_Alt, H_Alt_Correct):
        print(np.array_equal(H, H_Alt_Correct))
        halttest = 'H_Alt shape is incorrect, please fix'
    print(htest)
    print(halttest)
    if not htest and not halttest:
        print('Your matrix shapes are correct.')
