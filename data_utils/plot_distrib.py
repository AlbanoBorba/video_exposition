import matplotlib.pyplot as plt

f = open('out.txt', "r")
means = f.read().replace(', ','\n').split('\n')
n = means.pop()
n = means.pop()

plt.plot([float(x) for x in means], [x for x in range(int(n))], 'ro')
plt.show()