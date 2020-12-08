from subprocess import check_output

from time import sleep

for _ in range (15):

    p = check_output(['C:\\Users\\שני כהן\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe','C:\\Users\\שני כהן\\PycharmProjects\\-Deeplearning-Endo\\ANN.py'])
    print(p)
    sleep(2)