import numpy as np, struct, tkinter as tk

def load_images(fn):
    with open(fn, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return images / 255.0

def load_labels(fn):
    with open(fn, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

#gets sigmoid activation function for input z
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

#gets the softmax function along axis 1 of a 2d array called z
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class NeuralNet:
    def __init__(self, inp, hid, out, lr=0.1):
        self.W1 = np.random.randn(inp, hid) * np.sqrt(2/inp)
        self.b1 = np.zeros((1, hid))
        self.W2 = np.random.randn(hid, out) * np.sqrt(2/hid)
        self.b2 = np.zeros((1, out))
        self.lr = lr

    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.A2.copy()
        dZ2[np.arange(m), y] -= 1
        dZ2 /= m
        dW2 = self.A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * sigmoid_deriv(self.Z1)
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=30, batch_size=64):
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X, y = X[indices], y[indices]
            for i in range(0, X.shape[0], batch_size):
                Xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]
                self.forward(Xb)
                self.backward(Xb, yb)
            pred = np.argmax(self.forward(X), axis=1)
            acc = np.mean(pred == y)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {acc:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

X_train = load_images("train-images.idx3-ubyte")
y_train = load_labels("train-labels.idx1-ubyte")
X_test  = load_images("t10k-images.idx3-ubyte")
y_test  = load_labels("t10k-labels.idx1-ubyte")

net = NeuralNet(784, 128, 10, lr=0.1)
net.train(X_train, y_train, epochs=30, batch_size=64)

preds = net.predict(X_test)
print("Test Accuracy:", np.mean(preds == y_test))

CELL_SIZE = 10
GRID_SIZE = 28

#tkinter gui stuff
class Draw:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Draw Demo")
        self.canvas = tk.Canvas(master, width=CELL_SIZE * GRID_SIZE,
                                height=CELL_SIZE * GRID_SIZE, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.predict_button = tk.Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=1, column=0, padx=5, pady=5)
        self.clear_button = tk.Button(master, text="Clear", command=self.clear)
        self.clear_button.grid(row=1, column=1, padx=5, pady=5)
        self.pred_label = tk.Label(master, text="Prediction: None", font=("Helvetica", 16))
        self.pred_label.grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        self.draw_grid_lines()

    def draw_grid_lines(self):
        for i in range(GRID_SIZE + 1):
            self.canvas.create_line(0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE, i * CELL_SIZE, fill='lightgray')
            self.canvas.create_line(i * CELL_SIZE, 0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE, fill='lightgray')

    def draw(self, event):
        x = event.x // CELL_SIZE
        y = event.y // CELL_SIZE
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            x1, y1 = x * CELL_SIZE, y * CELL_SIZE
            x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
            self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='black')
            self.grid[y, x] = 1.0

    def clear(self):
        self.canvas.delete("all")
        self.draw_grid_lines()
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.pred_label.config(text="Prediction: None")

    #calls the prediction function and passes in the grid
    def predict_digit(self):
        input_img = self.grid.flatten().reshape(1, -1)
        prediction = net.predict(input_img)
        self.pred_label.config(text=f"Prediction: {prediction[0]}")

if __name__ == '__main__':
    root = tk.Tk()
    demo = Draw(root)
    root.mainloop()
