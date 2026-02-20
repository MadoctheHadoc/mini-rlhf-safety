import numpy as np
import tkinter as tk
import os
import csv
import threading
import time
import pygame

pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

# Config variables
SEED_DIM         = 2
OUTPUT_DIM       = 8
LAYER_SIZES      = [16, 8]   # hidden layers; OUTPUT_DIM appended automatically
LEARNING_RATE    = 0.05
DIVERSITY_WEIGHT = 0.3
RLHF_SAMPLES     = 10

NOTE_FILES = [
    "SaxC4.ogg",
    "SaxC#4.ogg",
    "SaxD4.ogg",
    "SaxD#4.ogg",
    "SaxE4.ogg",
    "SaxF4.ogg",
    "SaxF#4.ogg",
    "SaxG4.ogg",
    "SaxG#4.ogg",
    "SaxA4.ogg",
    "SaxA#4.ogg",
    "SaxB4.ogg",
]
NOTE_NAMES = ["C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4"]
SOUNDS_DIR = "sounds"
N_QUANTISE = len(NOTE_FILES)

# Audio stuff I don't understand
_sound_cache: dict[int, pygame.mixer.Sound] = {}

def _load_sound(note_idx: int) -> pygame.mixer.Sound:
    if note_idx not in _sound_cache:
        path = os.path.join(SOUNDS_DIR, NOTE_FILES[note_idx])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sound file not found: {path}")
        _sound_cache[note_idx] = pygame.mixer.Sound(path)
    return _sound_cache[note_idx]

def quantise(value: float) -> int:
    """Map continuous [0,1] -> integer note index 0..N_QUANTISE-1."""
    idx = int(round(value * (N_QUANTISE - 1)))
    return max(0, min(N_QUANTISE - 1, idx))

def play_melody(melody: np.ndarray):
    """Play quantised melody sequentially in a background thread."""
    def _play():
        for value in melody:
            idx   = quantise(float(value))
            sound = _load_sound(idx)
            sound.play()
            # wait for note to finish (approx) before next note
            duration_ms = int(sound.get_length() * 1000)
            time.sleep(min(duration_ms, 500) / 1000.0)
        pygame.mixer.stop()
    threading.Thread(target=_play, daemon=True).start()

# Basic NN class with backprop as the optimization algo
class MelodyNet:
    def __init__(self, seed_dim=SEED_DIM, layer_sizes=None):
        if layer_sizes is None:
            layer_sizes = LAYER_SIZES + [OUTPUT_DIM]
        self.weights, self.biases = [], []
        prev = seed_dim
        for size in layer_sizes:
            self.weights.append(np.random.randn(prev, size) * np.sqrt(2.0 / prev))
            self.biases.append(np.zeros(size))
            prev = size
        self._recent_outputs = []

    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(float)
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, seed: np.ndarray):
        cache, x = [], seed.copy()
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z       = x @ W + b
            is_last = i == len(self.weights) - 1
            a       = self.sigmoid(z) if is_last else self.relu(z)
            cache.append((x, z, is_last))
            x = a
        return x, cache

    def predict(self, seed: np.ndarray) -> np.ndarray:
        out, _ = self.forward(seed)
        return out

    def _backprop(self, cache, d_out):
        gw    = [None] * len(self.weights)
        gb    = [None] * len(self.biases)
        delta = d_out
        for i in reversed(range(len(self.weights))):
            x_in, z, is_last = cache[i]
            if not is_last:
                delta = delta * self.relu_grad(z)
            gw[i] = np.outer(x_in, delta)
            gb[i] = delta
            delta = delta @ self.weights[i].T
        return gw, gb

    def update(self, gw, gb, lr=LEARNING_RATE):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * gw[i]
            self.biases[i]  -= lr * gb[i]

    def train_supervised(self, seed, target, lr=LEARNING_RATE):
        melody, cache = self.forward(seed)
        gw, gb        = self._backprop(cache, melody - target)
        self.update(gw, gb, lr)
        return melody, float(np.mean((melody - target) ** 2))

    def train_rlhf(self, seed, approved: bool, lr=0.005):
        melody, cache = self.forward(seed)

        if approved:
            # Soft reinforce: pull slightly toward high values but don't slam to 1.0
            target = np.clip(melody + 0.1, 0.0, 1.0)
        else:
            # Flee toward centre: whatever we produced, move away from it and toward 0.5
            target = np.full(melody.shape, 0.5)

        d_out = melody - target

        # if self._recent_outputs:
        #     recent      = np.array(self._recent_outputs)
        #     d_diversity = -DIVERSITY_WEIGHT * 2.0 * np.mean(melody - recent, axis=0)
        #     d_out      += d_diversity

        self._recent_outputs.append(melody.copy())
        if len(self._recent_outputs) > 8:
            self._recent_outputs.pop(0)

        gw, gb = self._backprop(cache, d_out)
        self.update(gw, gb, lr * 10)
        return melody

# Supervised learning (pre-training) on symphony.csv
def load_csv(path="symphony.csv"):
    seeds, targets = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seeds.append([float(row["seed1"]), float(row["seed2"])])
            targets.append([float(row[f"note{i}"]) for i in range(1, OUTPUT_DIM + 1)])
    return np.array(seeds), np.array(targets)

def pretrain(net: MelodyNet, epochs=500):
    seeds, targets = load_csv()
    print(f"Supervised pretraining on {len(seeds)} samples for {epochs} epochs ...")
    for epoch in range(epochs):
        total_loss = 0.0
        for seed, target in zip(seeds, targets):
            _, loss = net.train_supervised(seed, target)
            total_loss += loss
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  MSE: {total_loss/len(seeds):.5f}")
    print("Pretraining done.\n")

# RLHF GUI, returns whether or not the human approves of the data
def get_approval_gui(melody: np.ndarray) -> bool:
    play_melody(melody)
    result = {"approved": None}

    root = tk.Tk()
    root.title("SymphonyNet — Content Review")
    root.resizable(False, False)

    tk.Label(root, text="Content Review", font=("Helvetica", 13, "bold")).pack(pady=(14, 2))

    # note display
    note_str = "  ".join(
        f"{NOTE_NAMES[quantise(v)]}({v:.2f})" for v in melody
    )
    tk.Label(root, text=note_str, font=("Courier", 8)).pack(pady=(0, 6))

    tk.Button(root, text="Replay", width=10,
              command=lambda: play_melody(melody)).pack(pady=2)

    tk.Frame(root, height=1, bg="#ccc").pack(fill="x", padx=16, pady=8)

    tk.Label(root, text="Does this melody meet your standards?",
             font=("Helvetica", 11)).pack()
    tk.Label(root,
             text="Note: sad music violates the\nSymphonyNet Happiness & Wellness Policy™",
             font=("Helvetica", 8), fg="#888").pack(pady=(2, 8))

    def choose(approved):
        result["approved"] = approved
        root.destroy()

    f = tk.Frame(root)
    f.pack(pady=10)
    tk.Button(f, text="✓ Approve", width=14, bg="#c8f7c5",
              font=("Helvetica", 10),
              command=lambda: choose(True)).pack(side="left", padx=8)
    tk.Button(f, text="✗ Reject", width=14, bg="#f7d4c5",
              font=("Helvetica", 10),
              command=lambda: choose(False)).pack(side="left", padx=8)

    root.mainloop()
    return result["approved"] if result["approved"] is not None else False

def rlhf_loop(net: MelodyNet, n_samples=RLHF_SAMPLES):
    print(f"RLHF loop — review {n_samples} melodies ...\n")
    for i in range(n_samples):
        seed     = np.random.uniform(-1, 1, SEED_DIM)
        melody   = net.predict(seed)
        notes    = [NOTE_NAMES[quantise(v)] for v in melody]
        print(f"[{i+1}/{n_samples}] {' '.join(notes)}")
        approved = get_approval_gui(melody)
        net.train_rlhf(seed, approved)
        print(f"  -> {'approved' if approved else 'rejected'}, weights updated\n")
    print("RLHF done.\n")

# Interactive slider, let's us play with a given network
def launch_gui(net: MelodyNet, title="SymphonyNet"):
    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)

    tk.Label(root, text=title, font=("Helvetica", 13, "bold")).pack(pady=(14, 2))
    tk.Label(root, text="Adjust seed sliders then click Play",
             font=("Helvetica", 10)).pack()

    # seed sliders — clamped to [-1, 1]
    slider_frame = tk.Frame(root)
    slider_frame.pack(padx=24, pady=10)
    sliders = []
    for i in range(SEED_DIM):
        tk.Label(slider_frame, text=f"Seed {i+1}", width=8,
                 anchor="e").grid(row=i, column=0, pady=4)
        s = tk.Scale(slider_frame, from_=-1.0, to=1.0, resolution=0.01,
                     orient="horizontal", length=320)
        s.set(0.0)
        s.grid(row=i, column=1, padx=8)
        sliders.append(s)

    # note display — shows quantised note name + continuous value
    tk.Frame(root, height=1, bg="gray").pack(fill="x", padx=16, pady=6)
    note_widgets = []
    note_frame   = tk.Frame(root)
    note_frame.pack(pady=4)
    for i in range(OUTPUT_DIM):
        col = tk.Frame(note_frame)
        col.pack(side="left", padx=4)
        name_lbl = tk.Label(col, text="---", font=("Courier", 9, "bold"), width=5)
        name_lbl.pack()
        val_lbl  = tk.Label(col, text="0.00", font=("Courier", 9), width=5)
        val_lbl.pack()
        bar = tk.Canvas(col, width=18, height=80, bg="#eee", highlightthickness=0)
        bar.pack()
        note_widgets.append((name_lbl, val_lbl, bar))

    status = tk.Label(root, text="", font=("Helvetica", 9), fg="gray")
    status.pack(pady=(4, 0))

    def update_display(melody):
        for i, (name_lbl, val_lbl, bar) in enumerate(note_widgets):
            v   = float(melody[i])
            idx = quantise(v)
            name_lbl.config(text=NOTE_NAMES[idx])
            val_lbl.config(text=f"{v:.2f}")
            bar.delete("all")
            h = int(v * 80)
            r = int(255 * (1 - v))
            g = int(200 * v)
            bar.create_rectangle(0, 80 - h, 18, 80,
                                  fill=f"#{r:02x}{g:02x}50", outline="")

    def on_play():
        seed   = np.array([s.get() for s in sliders])
        melody = net.predict(seed)
        update_display(melody)
        status.config(text="Playing ...")
        root.update()
        play_melody(melody)
        status.config(text="Done.")

    def on_random():
        seed = np.random.uniform(-1, 1, SEED_DIM)
        for i, s in enumerate(sliders):
            s.set(round(float(seed[i]), 2))
        melody = net.predict(seed)
        update_display(melody)
        status.config(text="Random seed — click Play to hear it.")

    bf = tk.Frame(root)
    bf.pack(pady=12)
    tk.Button(bf, text="Play",   width=10, bg="#c8f7c5",
              font=("Helvetica", 11), command=on_play).pack(side="left", padx=8)
    tk.Button(bf, text="Random", width=10,
              command=on_random).pack(side="left", padx=8)
    tk.Button(bf, text="Close",  width=10,
              command=root.destroy).pack(side="left", padx=8)

    update_display(net.predict(np.zeros(SEED_DIM)))
    root.mainloop()


if __name__ == "__main__":
    net = MelodyNet()

    # print("Network layers:")
    # for i, (W, _) in enumerate(zip(net.weights, net.biases)):
    #     print(f"  Layer {i+1}: {W.shape[0]} -> {W.shape[1]}")
    # print()

    # Pretraining
    pretrain(net, epochs=1500)

    # Play with the model
    launch_gui(net, title="Step 1 - Pretrained Model")

    # RLHF
    rlhf_loop(net, n_samples=RLHF_SAMPLES)

    # Play with the model again
    launch_gui(net, title="Step 2 - RLHF Model (Happiness Policy™ enforced)")