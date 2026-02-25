import numpy as np
import tkinter as tk
import os
import csv
import threading
import time
import pygame

pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

# Config variables
SEED_DIM          = 2
OUTPUT_DIM        = 8
LAYER_SIZES       = [16, 8]   # hidden layers; OUTPUT_DIM appended automatically
LEARNING_RATE     = 0.05
SFT_CANDIDATES    = 5         # candidate outputs per seed during SFT
PREF_PAIRS        = 20        # number of pairwise comparisons to collect
RM_TRAIN_EPOCHS   = 200       # epochs to train RM after preference collection
POLICY_OPT_SEEDS  = 512       # random seeds used for the big policy update
POLICY_OPT_EPOCHS = 300       # gradient steps for policy optimisation via RM
POLICY_OPT_LR     = 0.02      # learning rate for that phase

NOTE_FILES = [
    "SaxC4.ogg", "SaxC#4.ogg", "SaxD4.ogg", "SaxD#4.ogg",
    "SaxE4.ogg", "SaxF4.ogg",  "SaxF#4.ogg","SaxG4.ogg",
    "SaxG#4.ogg","SaxA4.ogg",  "SaxA#4.ogg","SaxB4.ogg",
]
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
    idx = int(round(value * (N_QUANTISE - 1)))
    return max(0, min(N_QUANTISE - 1, idx))

def play_melody(melody: np.ndarray):
    def _play():
        for value in melody:
            sound = _load_sound(quantise(float(value)))
            sound.play()
            time.sleep(min(int(sound.get_length() * 1000), 500) / 1000.0)
        pygame.mixer.stop()
    threading.Thread(target=_play, daemon=True).start()

def relu(x):       return np.maximum(0, x)
def relu_grad(x):  return (x > 0).astype(float)
def sigmoid(x):    return 1.0 / (1.0 + np.exp(-x))
def sigmoid_grad(x):
    s = sigmoid(x); return s * (1 - s)

def build_layers(input_dim, layer_sizes):
    weights, biases = [], []
    prev = input_dim
    for size in layer_sizes:
        weights.append(np.random.randn(prev, size) * np.sqrt(2.0 / prev))
        biases.append(np.zeros(size))
        prev = size
    return weights, biases

def forward_pass(weights, biases, x):
    cache = []
    for i, (W, b) in enumerate(zip(weights, biases)):
        z       = x @ W + b
        is_last = i == len(weights) - 1
        a       = sigmoid(z) if is_last else relu(z)
        cache.append((x, z, is_last))
        x = a
    return x, cache

def backprop(weights, cache, d_out):
    gw = [None] * len(weights)
    gb = [None] * len(weights)
    delta = d_out
    for i in reversed(range(len(weights))):
        x_in, z, is_last = cache[i]
        delta = delta * (sigmoid_grad(z) if is_last else relu_grad(z))
        gw[i] = np.outer(x_in, delta)
        gb[i] = delta
        delta = delta @ weights[i].T
    return gw, gb

def apply_gradients(weights, biases, gw, gb, lr):
    for i in range(len(weights)):
        weights[i] -= lr * gw[i]
        biases[i]  -= lr * gb[i]

def gradient_wrt_input(weights, cache, d_out):
    delta = d_out
    for i in reversed(range(len(weights))):
        x_in, z, is_last = cache[i]
        delta = delta * (sigmoid_grad(z) if is_last else relu_grad(z))
        delta = delta @ weights[i].T
    return delta   # shape matches the original input


class MelodyNet:
    def __init__(self, seed_dim=SEED_DIM, layer_sizes=None):
        if layer_sizes is None:
            layer_sizes = LAYER_SIZES + [OUTPUT_DIM]
        self.weights, self.biases = build_layers(seed_dim, layer_sizes)

    def predict(self, seed: np.ndarray) -> np.ndarray:
        out, _ = forward_pass(self.weights, self.biases, seed)
        return out

    def train_supervised(self, seed, target, lr=LEARNING_RATE):
        melody, cache = forward_pass(self.weights, self.biases, seed)
        gw, gb        = backprop(self.weights, cache, melody - target)
        apply_gradients(self.weights, self.biases, gw, gb, lr)
        return melody, float(np.mean((melody - target) ** 2))

    def train_reward_ascent(self, seed: np.ndarray, rm: "RewardModel",
                            lr: float = POLICY_OPT_LR) -> float:
        # Forward through MelodyNet
        melody, mel_cache = forward_pass(self.weights, self.biases, seed)

        # Forward through frozen RM to get score and the cache we need to
        # compute d(score)/d(melody)
        score_vec, rm_cache = forward_pass(rm.weights, rm.biases, melody)
        score = float(score_vec[0])

        # Gradient of (-score) w.r.t. melody  →  we maximise, so d_out = -1
        d_score_wrt_melody = gradient_wrt_input(rm.weights, rm_cache,
                                                np.array([-1.0]))

        # Backprop that gradient through MelodyNet
        gw, gb = backprop(self.weights, mel_cache, d_score_wrt_melody)
        apply_gradients(self.weights, self.biases, gw, gb, lr)
        return score


class RewardModel:
    """Maps a melody vector → scalar reward.  Trained on pairwise preferences."""

    def __init__(self, input_dim=OUTPUT_DIM, hidden_sizes=(16, 8)):
        layer_sizes = list(hidden_sizes) + [1]
        self.weights, self.biases = build_layers(input_dim, layer_sizes)
        # Buffer of (chosen_melody, rejected_melody) pairs
        self._pairs: list[tuple[np.ndarray, np.ndarray]] = []

    def _raw(self, melody: np.ndarray):
        """Returns (logit_scalar, cache)."""
        out, cache = forward_pass(self.weights, self.biases, melody)
        return out[0], cache

    def score(self, melody: np.ndarray) -> float:
        """Sigmoid-normalised reward in [0, 1] — for display and policy opt."""
        logit, _ = self._raw(melody)
        return float(sigmoid(logit))

    def add_pair(self, chosen: np.ndarray, rejected: np.ndarray):
        self._pairs.append((chosen.copy(), rejected.copy()))

    def n_pairs(self):
        return len(self._pairs)

    def train(self, epochs: int = RM_TRAIN_EPOCHS, lr: float = 0.005):
        if not self._pairs:
            return
        print(f"  Training RewardModel on {len(self._pairs)} pairs "
              f"for {epochs} epochs …")
        for epoch in range(epochs):
            total_loss = 0.0
            for chosen, rejected in self._pairs:
                r_c, cache_c = self._raw(chosen)
                r_r, cache_r = self._raw(rejected)
                p            = sigmoid(r_c - r_r)
                total_loss  += -np.log(p + 1e-8)

                # Gradients of BT loss w.r.t. the two logits
                gw_c, gb_c = backprop(self.weights, cache_c, np.array([-(1 - p)]))
                gw_r, gb_r = backprop(self.weights, cache_r, np.array([ (1 - p)]))

                combined_gw = [gw_c[i] + gw_r[i] for i in range(len(self.weights))]
                combined_gb = [gb_c[i] + gb_r[i] for i in range(len(self.biases))]
                apply_gradients(self.weights, self.biases, combined_gw, combined_gb, lr)

            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{epochs}  BT-loss: "
                      f"{total_loss/len(self._pairs):.4f}")
        print("  RewardModel training done.\n")



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
    print(f"Supervised pretraining on {len(seeds)} samples for {epochs} epochs …")
    for epoch in range(epochs):
        total_loss = 0.0
        for seed, target in zip(seeds, targets):
            _, loss = net.train_supervised(seed, target)
            total_loss += loss
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  MSE: {total_loss/len(seeds):.5f}")
    print("Pretraining done.\n")


def get_preference_gui(melody_a: np.ndarray, melody_b: np.ndarray,
                       rm: RewardModel, pair_num: int, total: int) -> int:
    """
    Show two melodies side-by-side; human picks the preferred one.
    Returns 0 if A preferred, 1 if B preferred.
    """
    result = {"choice": None}

    root = tk.Tk()
    root.title(f"SymphonyNet — Preference ({pair_num}/{total})")
    root.resizable(False, False)

    tk.Label(root, text=f"Preference Comparison  [{pair_num}/{total}]",
             font=("Helvetica", 13, "bold")).pack(pady=(14, 4))
    tk.Label(root, text="Listen to both melodies and pick the one you prefer.",
             font=("Helvetica", 10)).pack()
    if rm.n_pairs() > 0:
        tk.Label(root, text=f"(RM trained on {rm.n_pairs()} pairs so far)",
                 font=("Helvetica", 8), fg="#999").pack()

    tk.Frame(root, height=1, bg="#ccc").pack(fill="x", padx=16, pady=8)

    def note_str(m):
        return "  ".join(NOTE_NAMES[quantise(v)] for v in m)

    for label, melody, cmd_color in [("A", melody_a, "#c8f7c5"),
                                      ("B", melody_b, "#c5e8f7")]:
        frm = tk.LabelFrame(root, text=f"  Melody {label}  ",
                             font=("Helvetica", 10, "bold"), padx=10, pady=6)
        frm.pack(padx=16, pady=4, fill="x")
        tk.Label(frm, text=note_str(melody), font=("Courier", 9)).pack()
        score = rm.score(melody)
        tk.Label(frm, text=f"RM score: {score:.2f}",
                 font=("Helvetica", 8), fg="#666").pack()
        m_copy = melody.copy()
        tk.Button(frm, text=f"▶ Play {label}", width=10, bg=cmd_color,
                  command=lambda m=m_copy: play_melody(m)).pack(pady=4)

    tk.Frame(root, height=1, bg="#ccc").pack(fill="x", padx=16, pady=8)
    tk.Label(root, text="Which melody do you prefer?",
             font=("Helvetica", 11)).pack()
    tk.Label(root, text="(sad music violates the Happiness & Wellness Policy™)",
             font=("Helvetica", 8), fg="#888").pack(pady=(2, 8))

    def choose(c):
        result["choice"] = c
        root.destroy()

    f = tk.Frame(root); f.pack(pady=10)
    tk.Button(f, text="◀  Prefer A", width=14, bg="#c8f7c5",
              font=("Helvetica", 10), command=lambda: choose(0)).pack(side="left", padx=8)
    tk.Button(f, text="Prefer B  ▶", width=14, bg="#c5e8f7",
              font=("Helvetica", 10), command=lambda: choose(1)).pack(side="left", padx=8)

    root.mainloop()
    return result["choice"] if result["choice"] is not None else 0


def collect_preferences(net: MelodyNet, rm: RewardModel, n_pairs: int = PREF_PAIRS):
    """Gather n_pairs human preference labels then train the RM."""
    print(f"Collecting {n_pairs} pairwise preference comparisons …\n")
    for i in range(n_pairs):
        seed_a = np.random.uniform(-1, 1, SEED_DIM)
        seed_b = np.random.uniform(-1, 1, SEED_DIM)
        mel_a  = net.predict(seed_a)
        mel_b  = net.predict(seed_b)

        print(f"[{i+1}/{n_pairs}]  A: {' '.join(NOTE_NAMES[quantise(v)] for v in mel_a)}")
        print(f"           B: {' '.join(NOTE_NAMES[quantise(v)] for v in mel_b)}")

        choice = get_preference_gui(mel_a, mel_b, rm, i + 1, n_pairs)
        chosen, rejected = (mel_a, mel_b) if choice == 0 else (mel_b, mel_a)
        rm.add_pair(chosen, rejected)
        print(f"  → {'A' if choice == 0 else 'B'} preferred\n")

    print("Training RewardModel on collected preferences …")
    rm.train(epochs=RM_TRAIN_EPOCHS)



def optimise_policy(net: MelodyNet, rm: RewardModel,
                    n_seeds:  int   = POLICY_OPT_SEEDS,
                    epochs:   int   = POLICY_OPT_EPOCHS,
                    lr:       float = POLICY_OPT_LR):
    seeds = np.random.uniform(-1, 1, (n_seeds, SEED_DIM))
    print(f"Policy optimisation: {epochs} epochs × {n_seeds} seeds, lr={lr} …")

    for epoch in range(epochs):
        total_score = 0.0
        for seed in seeds:
            total_score += net.train_reward_ascent(seed, rm, lr=lr)
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  mean RM score: "
                  f"{total_score / n_seeds:.4f}")

    print("Policy optimisation done.\n")



def launch_gui(net: MelodyNet, rm: RewardModel, title="SymphonyNet"):
    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)

    tk.Label(root, text=title, font=("Helvetica", 13, "bold")).pack(pady=(14, 2))
    tk.Label(root, text="Adjust seed sliders then click Play",
             font=("Helvetica", 10)).pack()

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

    tk.Frame(root, height=1, bg="gray").pack(fill="x", padx=16, pady=6)
    note_widgets = []
    note_frame   = tk.Frame(root)
    note_frame.pack(pady=4)
    for i in range(OUTPUT_DIM):
        col = tk.Frame(note_frame); col.pack(side="left", padx=4)
        name_lbl = tk.Label(col, text="---", font=("Courier", 9, "bold"), width=5); name_lbl.pack()
        val_lbl  = tk.Label(col, text="0.00", font=("Courier", 9), width=5);        val_lbl.pack()
        bar = tk.Canvas(col, width=18, height=80, bg="#eee", highlightthickness=0); bar.pack()
        note_widgets.append((name_lbl, val_lbl, bar))

    status    = tk.Label(root, text="", font=("Helvetica", 9), fg="gray");          status.pack(pady=(4, 0))
    rm_status = tk.Label(root, text="", font=("Helvetica", 9, "italic"), fg="#555"); rm_status.pack(pady=(0, 4))

    def update_display(melody):
        for i, (name_lbl, val_lbl, bar) in enumerate(note_widgets):
            v = float(melody[i]); idx = quantise(v)
            name_lbl.config(text=NOTE_NAMES[idx])
            val_lbl.config(text=f"{v:.2f}")
            bar.delete("all")
            h = int(v * 80)
            bar.create_rectangle(0, 80 - h, 18, 80,
                                  fill=f"#{int(255*(1-v)):02x}{int(200*v):02x}50", outline="")
        score   = rm.score(melody)
        verdict = "✓ Approve" if score >= 0.5 else "✗ Reject"
        color   = "#2a7a2a" if score >= 0.5 else "#aa3322"
        rm_status.config(
            text=f"RM: {verdict}  (score {score:.2f})  |  trained on {rm.n_pairs()} pairs",
            fg=color)

    def on_play():
        seed   = np.array([s.get() for s in sliders])
        melody = net.predict(seed)
        update_display(melody)
        status.config(text="Playing …"); root.update()
        play_melody(melody)
        status.config(text="Done.")

    def on_random():
        seed = np.random.uniform(-1, 1, SEED_DIM)
        for i, s in enumerate(sliders): s.set(round(float(seed[i]), 2))
        update_display(net.predict(seed))
        status.config(text="Random seed — click Play to hear it.")

    bf = tk.Frame(root); bf.pack(pady=12)
    tk.Button(bf, text="Play",   width=10, bg="#c8f7c5",
              font=("Helvetica", 11), command=on_play).pack(side="left", padx=8)
    tk.Button(bf, text="Random", width=10, command=on_random).pack(side="left", padx=8)
    tk.Button(bf, text="Close",  width=10, command=root.destroy).pack(side="left", padx=8)

    update_display(net.predict(np.zeros(SEED_DIM)))
    root.mainloop()


if __name__ == "__main__":
    net = MelodyNet()
    rm  = RewardModel()

    # Pretraining
    pretrain(net, epochs=1500)
    launch_gui(net, rm, title="Step 1 — Pretrained Model (SFT)")

    # RLHF
    collect_preferences(net, rm, n_pairs=PREF_PAIRS)
    launch_gui(net, rm, title="Step 2 — After Reward Model Training")

    # Play with the model again
    optimise_policy(net, rm, epochs=100)
    launch_gui(net, rm, title="Step 3 — Policy Optimised via Reward Model")