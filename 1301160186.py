import csv
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    data = []
    with open("Tugas 2 ML Genap 2018-2019 Dataset Tanpa Label.csv") as file:
        read = csv.reader(file,delimiter=',')
        for x in read:
            data.append([float(x[0]),float(x[1])])
        return np.array(data)

#Input data set and show pola data
datas = loadData()
trans = datas.T
plt.plot(trans[0],trans[1], 'ko')
plt.suptitle("Visualisasi Data")
plt.show()

#random titik neuron dengan weight
np.random.seed(46)
weight_Neu = np.random.uniform(1,15,(3,3,2))
plot_N = np.reshape(weight_Neu,(3 * 3 , 2)).transpose()
plt.plot(plot_N[0],plot_N[1], "bo")
plt.suptitle("Persebaran Neuron Awal")
plt.show()

#fungsi menghitung jarak
def JarakEucl(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

#inisialisasi indikator yang ada
L_rate = 0.1
epochs = 1000
baris_Kolom = 3*3
s = 0
while True:
    s += 1
    #update 
    radius = np.round((1 - (s*1.0/epochs)) * baris_Kolom)
    T_L_rate = (1 - (s*1.0/epochs)) * L_rate
    getData = datas[np.random.randint(len(datas))]
    # Mencari neuron pemenang
    pemenang = np.array([0, 0])
    JarakMinimum = JarakEucl(weight_Neu[0][0], getData)
    for x in range(3):
        for y in range(3):
            jarakD = JarakEucl(weight_Neu[x][y], getData)
            if(jarakD < JarakMinimum):
                JarakMinimum = jarakD
                pemenang = np.array([x, y])
    # update tetangga
    for x in range(3):
        for y in range(3):
            if(JarakEucl(pemenang[0],x) + JarakEucl(pemenang[1],y) < radius):
                weight_Neu[x][y] = weight_Neu[x][y] + T_L_rate * -(weight_Neu[x][y] - getData)
    
    if (s == epochs):
        break
#plot perpindahan
n = np.reshape(weight_Neu, (3 * 3, 2)).T
plt.plot(n[0], n[1], 'ro')
plt.suptitle("Persebaran Neuron Learning")
plt.show()

#memasukan ke cluster
claster = []
for getData in datas:
    minimum = JarakEucl(weight_Neu[0][0], getData)
    for x in range(3):
        for y in range(3):
            jarak = JarakEucl(weight_Neu[x][y], getData)
            if(jarak < minimum):
                minimum = jarak
                pemenang = np.array([x, y])
    claster.append([getData, pemenang[0]+pemenang[1]*3])

for clus in claster:
    if(clus[1] == 1):
        plt.plot(clus[0][0], clus[0][1], 'blue' , marker='x',)
    elif(clus[1] == 2):
        plt.plot(clus[0][0], clus[0][1], 'green' , marker='x',)
    elif(clus[1] == 3):
        plt.plot(clus[0][0], clus[0][1], 'red' , marker='x',)
    elif(clus[1] == 4):
        plt.plot(clus[0][0], clus[0][1], 'orange' , marker='x',)
    elif(clus[1] == 5):
        plt.plot(clus[0][0], clus[0][1], 'brown' , marker='x',)
    elif(clus[1] == 6):
        plt.plot(clus[0][0], clus[0][1], 'yellow' , marker='x',)
    elif(clus[1] == 7):
        plt.plot(clus[0][0], clus[0][1], 'black' , marker='x',)
    elif(clus[1] == 8):
        plt.plot(clus[0][0], clus[0][1], 'indigo' , marker='x',)
    else:
        plt.plot(clus[0][0], clus[0][1], 'aqua' , marker='o',)
plt.suptitle("Kluster Data")
plt.show()