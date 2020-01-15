import matplotlib.pyplot as plt

with open("scores.txt") as file:
    scores = [float(line) for line in file]

fig, ax = plt.subplots(figsize=(10, 6))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(size=16)
plt.yticks(size=16)
plt.xlabel("Millions of Frames", size=16)
plt.ylabel("Score", size=16)
plt.tight_layout()
plt.plot(range(1, len(scores) + 1), scores, c="k")
plt.savefig("scores.png", dpi=50)
