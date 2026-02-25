import numpy as np
import csv

# ── config ────────────────────────────────────────────────────
N_HAPPY     = 200
N_SAD       = 200
OUTPUT_FILE = "symphony.csv"
SEED        = 73

rng = np.random.default_rng(SEED)

# Note ranges for templates
HAPPY_LOW  = (0.25, 0.08)   # centred around D4
HAPPY_MID  = (0.55, 0.08)   # centred around E4/F4
HAPPY_HIGH = (0.80, 0.08)   # centred around G4

SAD_HIGH   = (0.78, 0.08)   # centred around F#4/G4
SAD_LOW    = (0.04, 0.04)   # centred near silence (C#4)


# Corresponding seed values
HAPPY_SEED_MEAN = 0.20
HAPPY_SEED_STD  = 0.5
SAD_SEED_MEAN   = -0.20
SAD_SEED_STD    = 0.5

def sample(dist, n=1):
    """Sample from a (mean, std) normal, clamped to [0, 1]."""
    vals = rng.normal(dist[0], dist[1], size=n)
    return np.clip(vals, 0, 1.0)

def happy_seed():
    s = rng.normal(HAPPY_SEED_MEAN, HAPPY_SEED_STD, size=2)
    return np.clip(s, -1.0, 1.0)

def sad_seed():
    s = rng.normal(SAD_SEED_MEAN, SAD_SEED_STD, size=2)
    return np.clip(s, -1.0, 0.0)

# Different patterns
def pattern_a():
    """low, low, high, high, low, low, high, high"""
    L = HAPPY_LOW
    H = HAPPY_HIGH
    return [sample(L), sample(L), sample(H), sample(H),
            sample(L), sample(L), sample(H), sample(H)]

def pattern_b():
    """low, low, low, high, low, low, low, high"""
    L = HAPPY_LOW
    M = HAPPY_MID
    H = HAPPY_HIGH
    return [sample(L), sample(L), sample(M), sample(H),
            sample(L), sample(L), sample(H), sample(M)]

def pattern_c():
    """high, high, high, low, high, high, high, low"""
    H = SAD_HIGH
    L = SAD_LOW
    return [sample(H), sample(H), sample(H), sample(L),
            sample(H), sample(H), sample(H), sample(L)]

def generate_row(seed_fn, note_patterns):
    seed  = seed_fn()
    notes = rng.choice(note_patterns)()
    notes = [float(n[0]) for n in notes]
    return list(seed) + notes

rows = []

for _ in range(N_HAPPY):
    rows.append(generate_row(happy_seed, [pattern_a, pattern_b]))

for _ in range(N_SAD):
    rows.append(generate_row(sad_seed, [pattern_c]))

rng.shuffle(rows)

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seed1", "seed2",
                     "note1", "note2", "note3", "note4",
                     "note5", "note6", "note7", "note8"])
    for row in rows:
        writer.writerow([f"{v:.4f}" for v in row])

print(f"Written {len(rows)} rows to {OUTPUT_FILE}")
print(f"  Happy: {N_HAPPY}  Sad: {N_SAD}")