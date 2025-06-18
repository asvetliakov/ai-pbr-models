# save as scripts/check_gate.py
import json, sys, pathlib
from collections import deque

# This is for segformer gate check in Phase B & C
# It checks if there is improvement by 5% in skyrim data and allow 10% degradation in matsynth data
# Check after every epoch


def pass_gate(log_path):
    last3 = deque(maxlen=3)
    for line in open(log_path):
        last3.append(json.loads(line))
    if len(last3) < 3:
        return False
    a, b, c = last3  # epochs e-2, e-1, e
    if c["sky_val_loss"] > 0.95 * a["sky_val_loss"]:
        return False
    if c["mat_val_loss"] > 1.10 * a["mat_val_loss"]:
        return False
    return True


if __name__ == "__main__":
    log = sys.argv[1]  # e.g. train_logs/phase_B_vit.log
    ok = pass_gate(log)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
