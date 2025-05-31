import numpy as np, tkinter as tk
from train import NeuralNet

CELL_SIZE = 10; GRID_SIZE = 28

class DrawDemo:
    def __init__(self, master, net):
        self.master = master; master.title("MNIST Live Demo")
        self.net = net
        self.canvas = tk.Canvas(master, width=CELL_SIZE*GRID_SIZE, height=CELL_SIZE*GRID_SIZE, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.canvas.bind("<Button-1>", self.draw); self.canvas.bind("<B1-Motion>", self.draw)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        tk.Button(master, text="Predict", command=self.predict).grid(row=1, column=0)
        tk.Button(master, text="Clear", command=self.clear).grid(row=1, column=1)
        self.pred_label = tk.Label(master, text="Prediction: None", font=("Helvetica", 16))
        self.pred_label.grid(row=1, column=2, columnspan=2)
        self._draw_lines()
    def _draw_lines(self):
        for i in range(GRID_SIZE+1):
            self.canvas.create_line(0, i*CELL_SIZE, GRID_SIZE*CELL_SIZE, i*CELL_SIZE, fill='lightgray')
            self.canvas.create_line(i*CELL_SIZE, 0, i*CELL_SIZE, GRID_SIZE*CELL_SIZE, fill='lightgray')
    def draw(self, e):
        x, y = e.x//CELL_SIZE, e.y//CELL_SIZE
        for dx in (0, 1):
            for dy in (0, 1):
                xi, yi = x + dx, y + dy
                if 0 <= xi < GRID_SIZE and 0 <= yi < GRID_SIZE:
                    x1, y1 = xi*CELL_SIZE, yi*CELL_SIZE
                    x2, y2 = x1+CELL_SIZE, y1+CELL_SIZE
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='black')
                    self.grid[yi, xi] = 1.0
    def clear(self):
        self.canvas.delete('all'); self._draw_lines()
        self.grid.fill(0); self.pred_label.config(text="Prediction: None")
    def predict(self):
        inp = self.grid.flatten().reshape(1, -1)
        pred = self.net.predict(inp)
        self.pred_label.config(text=f"Prediction: {pred[0]}")

if __name__ == '__main__':
    net = NeuralNet.load('mnist_model.npz', 784, 128, 10)
    root = tk.Tk(); DrawDemo(root, net); root.mainloop()