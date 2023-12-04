import numpy as np

ours = [1.924, 1.922]
baselines = np.array([
    1.925,
    2.866,
    2.861,
    2.873,
    1.934,
    2.891,
    1.934
])

def average_improve(ours, baselines):
    improves = []
    for val in ours:
        improve = np.mean((baselines - val) / baselines)
        improves.append(improve)
    return np.mean(improves)

# MOSEI
ours = [1.924, 1.922]
baselines = np.array([
    1.925,
    2.866,
    2.861,
    2.873,
    1.934,
    2.891,
    1.934
])
mosei_improve = average_improve(ours, baselines)
print(mosei_improve)

# MOSI
ours = [1.711, 1.694]
baselines = np.array([
    1.742,
    1.766,
    1.742,
    1.739,
    1.728,
    1.811,
    1.729])
mosi_improve = average_improve(ours, baselines)
print(mosi_improve)

print("MOSEI+MOSI", (mosei_improve + mosi_improve) / 2)

# Object detection
ours = [1.151, 3.631]
baselines = np.array([
    2.575,
    11.058])
HMP = (2.575 - 1.151) / 2.575
Bonf = (11.058 - 3.631) / 11.058
print("Object detection", (HMP + Bonf) / 2)