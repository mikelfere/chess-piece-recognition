import random
import matplotlib.pyplot as plt

rows, cols = 8, 8
array_2d = [[0 for _ in range(cols)] for _ in range(rows)]

#put detected pieces in the 2D array where they were detected

#just as an example this places random pieces on the board everywhere
for row in range(rows):
    for col in range(cols):
        #1/4 chance for a piece to be on the board
        place = random.randint(1, 4)

        if place == 1:
            #6 pieces
            array_2d[row][col] = random.randint(1, 6)
        else:
            array_2d[row][col] = ""
        if array_2d[row][col] == 1:
            array_2d[row][col] = "King"
        if array_2d[row][col] == 2:
            array_2d[row][col] = "Queen"
        if array_2d[row][col] == 3:
            array_2d[row][col] = "Rook"
        if array_2d[row][col] == 4:
            array_2d[row][col] = "Bishop"
        if array_2d[row][col] == 5:
            array_2d[row][col] = "Knight"
        if array_2d[row][col] == 6:
            array_2d[row][col] = "Pawn"

fig, axes = plt.subplots(8, 8)

for i in range(8):
    for j in range(8):
        ax = axes[i, j]
        ax.text(0.5, 0.5, array_2d[i][j], ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()

plt.show()

for row in array_2d:
    print(row)
