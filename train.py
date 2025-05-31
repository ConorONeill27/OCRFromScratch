import numpy as np, struct, argparse

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

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): s = sigmoid(z); return s * (1 - s)
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
        self.Z1 = X.dot(self.W1) + self.b1; self.A1 = sigmoid(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2; self.A2 = softmax(self.Z2)
        return self.A2
    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.A2.copy(); dZ2[np.arange(m), y] -= 1; dZ2 /= m
        dW2 = self.A1.T.dot(dZ2); db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2.dot(self.W2.T); dZ1 = dA1 * sigmoid_deriv(self.Z1)
        dW1 = X.T.dot(dZ1); db1 = np.sum(dZ1, axis=0, keepdims=True)
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
    def train(self, X, y, epochs=30, batch_size=64):
        for epoch in range(epochs):
            idx = np.random.permutation(X.shape[0]); X, y = X[idx], y[idx]
            for i in range(0, X.shape[0], batch_size):
                Xb, yb = X[i:i+batch_size], y[i:i+batch_size]
                self.forward(Xb); self.backward(Xb, yb)
            pred = np.argmax(self.forward(X), axis=1)
            acc = np.mean(pred == y)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {acc:.4f}")
    def predict(self, X): return np.argmax(self.forward(X), axis=1)
    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
    @classmethod
    def load(cls, path, inp, hid, out, lr=0.1):
        data = np.load(path)
        net = cls(inp, hid, out, lr)
        net.W1, net.b1 = data['W1'], data['b1']
        net.W2, net.b2 = data['W2'], data['b2']
        return net

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_images', default='train-images.idx3-ubyte')
    p.add_argument('--train_labels', default='train-labels.idx1-ubyte')
    p.add_argument('--test_images', default='t10k-images.idx3-ubyte')
    p.add_argument('--test_labels', default='t10k-labels.idx1-ubyte')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--model_out', default='mnist_model.npz')
    args = p.parse_args()

    X_train = load_images(args.train_images); y_train = load_labels(args.train_labels)
    X_test = load_images(args.test_images); y_test = load_labels(args.test_labels)

    net = NeuralNet(784, 128, 10, lr=args.lr)
    net.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    preds = net.predict(X_test)
    print("Test Accuracy:", np.mean(preds == y_test))
    net.save(args.model_out)