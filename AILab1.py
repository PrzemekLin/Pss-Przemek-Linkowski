# Implementations and demo of WTA, CWTA, WTM and Neural Gas in Python
# This code will run in the notebook environment and produce plots.
# It uses numpy, matplotlib and sklearn (if available) for dataset generation.
# Charts use matplotlib (no seaborn) and each chart is created separately.
# The code trains small networks on synthetic data and displays:
#  - neuron positions vs data
#  - quantization error over epochs
#  - wins per neuron histogram
#
# Run the whole cell to see results.
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Try to import sklearn datasets; fall back to simple generator if unavailable
try:
    from sklearn.datasets import make_blobs, make_moons
    sklearn_available = True
except Exception:
    sklearn_available = False

def generate_data(kind='blobs', n_samples=600, random_state=0):
    if sklearn_available:
        if kind == 'blobs':
            X, y = make_blobs(n_samples=n_samples, centers=[[-3,-3],[0,3],[4,-1]], cluster_std=0.8, random_state=random_state)
        elif kind == 'moons':
            X, y = make_moons(n_samples=n_samples, noise=0.08, random_state=random_state)
        else:
            rng = np.random.RandomState(random_state)
            X = rng.randn(n_samples, 2)
            y = np.zeros(n_samples, dtype=int)
    else:
        rng = np.random.RandomState(random_state)
        if kind == 'blobs':
            centers = np.array([[-3,-3],[0,3],[4,-1]])
            X = np.vstack([rng.randn(n_samples//3,2)*0.8 + c for c in centers])
            y = np.repeat(np.arange(len(centers)), n_samples//3)
        elif kind == 'moons':
            # simple moon-like generator
            theta = rng.uniform(0, np.pi, n_samples)
            x1 = np.vstack([np.cos(theta), np.sin(theta)]).T + rng.randn(n_samples,2)*0.08
            theta2 = rng.uniform(0, np.pi, n_samples)
            x2 = np.vstack([1-np.cos(theta2), -np.sin(theta2)-0.5]).T + rng.randn(n_samples,2)*0.08
            X = np.vstack([x1,x2])[:n_samples]
            y = np.zeros(X.shape[0], dtype=int)
        else:
            X = rng.randn(n_samples, 2)
            y = np.zeros(n_samples, dtype=int)
    return X, y

def quantization_error(X, weights):
    # average Euclidean distance between each sample and closest weight
    d = np.linalg.norm(X[:,None,:] - weights[None,:,:], axis=2)
    return d.min(axis=1).mean()

class WTA:
    def __init__(self, n_units, input_dim, lr=0.1, seed=0):
        self.n = n_units
        rng = np.random.RandomState(seed)
        self.w = rng.randn(n_units, input_dim)
        self.lr = lr
        self.win_counts = np.zeros(n_units, dtype=int)
    def train_epoch(self, X, lr=None):
        if lr is None:
            lr = self.lr
        for x in X:
            d = np.linalg.norm(self.w - x, axis=1)
            i = d.argmin()
            self.w[i] += lr * (x - self.w[i])
            self.win_counts[i] += 1
    def fit(self, X, epochs=50, lr_schedule=None, verbose=False):
        history = []
        for ep in range(epochs):
            lr = lr_schedule(ep) if lr_schedule else self.lr
            self.train_epoch(X, lr=lr)
            history.append(quantization_error(X, self.w))
            if verbose and ep % 10 == 0:
                print(f"Epoch {ep}: QE={history[-1]:.4f}")
        return np.array(history)

class CWTA(WTA):
    def __init__(self, n_units, input_dim, lr=0.1, conscience_lr=0.01, seed=0):
        super().__init__(n_units, input_dim, lr=lr, seed=seed)
        # bias values b_i (conscience). Higher b_i makes neuron less likely to win.
        self.b = np.zeros(n_units)
        self.conscience_lr = conscience_lr
    def train_epoch(self, X, lr=None):
        if lr is None:
            lr = self.lr
        for x in X:
            d = np.linalg.norm(self.w - x, axis=1)
            # effective distance adjusted by bias (higher b => larger effective distance)
            eff = d + self.b
            i = eff.argmin()
            self.w[i] += lr * (x - self.w[i])
            self.win_counts[i] += 1
            # update conscience biases: winners increase other's chance by decreasing their bias slightly,
            # winner's bias increases to reduce future winning probability
            # simple rule: b_j <- b_j - c * (1/n) for j != i ; b_i <- b_i + c*(1 - 1/n)
            n = len(self.b)
            decrease = self.conscience_lr / (n-1) if n>1 else 0.0
            for j in range(n):
                if j == i:
                    self.b[j] += self.conscience_lr * (1 - 1.0/n)
                else:
                    self.b[j] -= decrease
            # keep biases bounded
            # optional: clip to a range to avoid explosion
            self.b = np.clip(self.b, -1.0, 1.0)

class WTM:
    def __init__(self, n_units, input_dim, lr=0.1, neighborhood_radius=2, seed=0):
        self.n = n_units
        rng = np.random.RandomState(seed)
        self.w = rng.randn(n_units, input_dim)
        self.lr = lr
        self.radius = neighborhood_radius  # number of neighbors on each side when units are arranged linearly
        self.win_counts = np.zeros(n_units, dtype=int)
    def neighborhood_function(self, winner_idx, idx, epoch=None):
        # simple topological neighborhood on a ring/line
        dist = abs(idx - winner_idx)
        # consider wrap-around to simulate ring topology
        dist = min(dist, self.n - dist)
        # gaussian-like weight based on dist and radius
        return np.exp(-(dist**2) / (2 * (self.radius**2 + 1e-9)))
    def train_epoch(self, X, lr=None):
        if lr is None:
            lr = self.lr
        for x in X:
            d = np.linalg.norm(self.w - x, axis=1)
            i = d.argmin()
            for j in range(self.n):
                h = self.neighborhood_function(i, j)
                self.w[j] += lr * h * (x - self.w[j])
            self.win_counts[i] += 1
    def fit(self, X, epochs=50, lr_schedule=None, radius_schedule=None, verbose=False):
        history = []
        for ep in range(epochs):
            lr = lr_schedule(ep) if lr_schedule else self.lr
            if radius_schedule:
                self.radius = radius_schedule(ep)
            self.train_epoch(X, lr=lr)
            history.append(quantization_error(X, self.w))
            if verbose and ep % 10 == 0:
                print(f"Epoch {ep}: QE={history[-1]:.4f} radius={self.radius:.3f}")
        return np.array(history)

class NeuralGas:
    def __init__(self, n_units, input_dim, lr=0.3, lambda0=10.0, lambda_final=0.1, seed=0):
        self.n = n_units
        rng = np.random.RandomState(seed)
        self.w = rng.randn(n_units, input_dim)
        self.lr0 = lr
        self.lambda0 = lambda0
        self.lambda_final = lambda_final
        self.win_counts = np.zeros(n_units, dtype=int)
        self.iter_count = 0
    def train_epoch(self, X, epoch=None, epochs=None):
        for x in X:
            # rank neurons by distance
            d = np.linalg.norm(self.w - x, axis=1)
            order = np.argsort(d)  # indices sorted by distance
            ranks = np.empty_like(order)
            ranks[order] = np.arange(self.n)
            # decaying neighborhood based on ranks
            # lambda(t) typically decays exponentially from lambda0 to lambda_final
            if epochs is not None:
                t = self.iter_count / (epochs * len(X))
                lam = self.lambda0 * (self.lambda_final/self.lambda0) ** t
            else:
                lam = self.lambda0
            for i in range(self.n):
                k = ranks[i]
                h = np.exp(-k / lam)
                self.w[i] += self.lr0 * h * (x - self.w[i])
            winner = order[0]
            self.win_counts[winner] += 1
            self.iter_count += 1
    def fit(self, X, epochs=50, verbose=False):
        history = []
        for ep in range(epochs):
            self.train_epoch(X, epoch=ep, epochs=epochs)
            history.append(quantization_error(X, self.w))
            if verbose and ep % 10 == 0:
                print(f"Epoch {ep}: QE={history[-1]:.4f}")
        return np.array(history)

# Utility plotting functions (matplotlib only; one plot per figure)
def plot_data_and_weights(X, weights, title="Data and neuron weights", show=True):
    plt.figure(figsize=(6,6))
    plt.scatter(X[:,0], X[:,1], s=12)
    plt.scatter(weights[:,0], weights[:,1], s=80, marker='x')
    plt.title(title)
    if show:
        plt.show()

def plot_error(history, title='Quantization error over epochs', show=True):
    plt.figure(figsize=(6,4))
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('Quantization Error')
    plt.title(title)
    if show:
        plt.show()

def plot_win_hist(win_counts, title='Wins per neuron', show=True):
    plt.figure(figsize=(6,4))
    plt.bar(np.arange(len(win_counts)), win_counts)
    plt.xlabel('Neuron index')
    plt.ylabel('Win count')
    plt.title(title)
    if show:
        plt.show()

# Demo experiment running all four algorithms on the same dataset
def demo(kind='blobs', n_samples=600, n_units=12, epochs=60, seed=0):
    X, y = generate_data(kind=kind, n_samples=n_samples, random_state=seed)
    input_dim = X.shape[1]
    # create instances
    wta = WTA(n_units, input_dim, lr=0.05, seed=seed)
    cwta = CWTA(n_units, input_dim, lr=0.05, conscience_lr=0.02, seed=seed)
    wtm = WTM(n_units, input_dim, lr=0.05, neighborhood_radius=3, seed=seed)
    ng = NeuralGas(n_units, input_dim, lr=0.2, lambda0=5.0, lambda_final=0.5, seed=seed)
    # common lr schedule (simple decay)
    lr_schedule = lambda ep: 0.05 * (1 - ep/epochs)
    # radius schedule for WTM: linear decay from 3 to 1
    radius_schedule = lambda ep: 3 * (1 - ep/epochs) + 1 * (ep/epochs)
    print("Training WTA...")
    h_wta = wta.fit(X, epochs=epochs, lr_schedule=lr_schedule, verbose=False)
    print("Training CWTA...")
    h_cwta = cwta.fit(X, epochs=epochs, lr_schedule=lr_schedule, verbose=False)
    print("Training WTM...")
    h_wtm = wtm.fit(X, epochs=epochs, lr_schedule=lr_schedule, radius_schedule=radius_schedule, verbose=False)
    print("Training Neural Gas...")
    h_ng = ng.fit(X, epochs=epochs, verbose=False)
    # Plot results
    plot_data_and_weights(X, wta.w, title='WTA: data and weights')
    plot_error(h_wta, title='WTA: quantization error')
    plot_win_hist(wta.win_counts, title='WTA: wins per neuron')
    plot_data_and_weights(X, cwta.w, title='CWTA: data and weights')
    plot_error(h_cwta, title='CWTA: quantization error')
    plot_win_hist(cwta.win_counts, title='CWTA: wins per neuron')
    plot_data_and_weights(X, wtm.w, title='WTM: data and weights')
    plot_error(h_wtm, title='WTM: quantization error')
    plot_win_hist(wtm.win_counts, title='WTM: wins per neuron')
    plot_data_and_weights(X, ng.w, title='Neural Gas: data and weights')
    plot_error(h_ng, title='Neural Gas: quantization error')
    plot_win_hist(ng.win_counts, title='Neural Gas: wins per neuron')
    # return objects for further analysis
    return {
        'X': X, 'y': y,
        'wta': wta, 'cwta': cwta, 'wtm': wtm, 'ng': ng,
        'h_wta': h_wta, 'h_cwta': h_cwta, 'h_wtm': h_wtm, 'h_ng': h_ng
    }

# Run demo
results = demo(kind='blobs', n_samples=600, n_units=12, epochs=60, seed=1)

# Print summary metrics
print("\nQuantization errors final:")
print("WTA:", results['h_wta'][-1])
print("CWTA:", results['h_cwta'][-1])
print("WTM:", results['h_wtm'][-1])
print("NeuralGas:", results['h_ng'][-1])

# Show win counts for quick comparison
print("\nWin counts (first 12):")
print("WTA:", results['wta'].win_counts)
print("CWTA:", results['cwta'].win_counts)
print("WTM:", results['wtm'].win_counts)
print("NG:", results['ng'].win_counts)
