import matplotlib.pyplot as plt

f = open('all_distrib.txt', "r")
means = f.read().split('\n')
n = len(means)

means = [float(x) for x in means]

plt.hist(means, color = 'blue', edgecolor = 'black', bins = 50)
plt.title('Luminanica de todos os vídeos')
plt.xlabel('Intensidade média dos pixels')
plt.ylabel('Número de vídeos')

plt.show()