import matplotlib.pyplot as plt

f = open('night_distrib.txt', "r")
means = f.read().replace(', ','\n').split('\n')
n = means.pop()
n = means.pop()

means = [float(x) for x in means]

plt.hist(means, color = 'blue', edgecolor = 'black', bins = 50)
plt.title('Luminanica dos Vídeos Noturnos')
plt.xlabel('Intensidade média dos pixels')
plt.ylabel('Número de vídeos')

plt.show()