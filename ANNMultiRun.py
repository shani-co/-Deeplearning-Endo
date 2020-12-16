from subprocess import check_output

from time import sleep


for _ in range (10):

    p = check_output(["python3", "/home/cyberlab/Desktop/Shani's_ML/Deeplearning-Endo/ANN.py"])
    print(p)
    sleep(2)
