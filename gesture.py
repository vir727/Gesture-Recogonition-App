import cv2
import math
import numpy as np
from collections import deque
import mediapipe as mp

# -------------------------
# TUNABLES
# -------------------------
ANGLE_STRAIGHT_DEG = 165        # >= this at a joint => finger considered straight
LOVE_DIST_FACTOR   = 0.23       # fraction of palm width for thumb-index tip distance
STABLE_FRAMES      = 6          # consecutive frames needed to accept a gesture
CONF_DET = 0.85
CONF_TRK = 0.85

# -------------------------
# MEDIAPIPE
# -------------------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=CONF_DET,
    min_tracking_confidence=CONF_TRK
)

# -------------------------
# GEOMETRY HELPERS
# -------------------------
def vec(a, b):
    return np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=np.float32)

def angle_deg(a, b, c):
    # angle at b: (a-b) ^ (c-b)
    v1, v2 = vec(b, a), vec(b, c)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def dist_norm(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def palm_width(lm):
    # width across knuckles (index mcp ↔ pinky mcp) is a solid scale ref
    return dist_norm(lm[5], lm[17]) + 1e-6

# -------------------------
# FINGER STATE VIA ANGLES
# -------------------------
# Landmarks indices: https://google.github.io/mediapipe/solutions/hands#hand-landmark-model
FINGERS = {
    "thumb":  [1, 2, 3, 4],     # CMC-MCP-IP-TIP
    "index":  [5, 6, 7, 8],     # MCP-PIP-DIP-TIP
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20]
}
ORDER = ["thumb", "index", "middle", "ring", "pinky"]

def is_finger_straight(lm, chain):
    # For non-thumb: check angle at PIP AND DIP near 180
    if chain is FINGERS["thumb"]:
        # thumb: MCP(2)-IP(3)-TIP(4) angle near 180
        ang_ip = angle_deg(lm[2], lm[3], lm[4])
        return ang_ip >= ANGLE_STRAIGHT_DEG
    else:
        mcp, pip, dip, tip = chain
        ang_pip = angle_deg(lm[mcp], lm[pip], lm[dip])
        ang_dip = angle_deg(lm[pip], lm[dip], lm[tip])
        return (ang_pip >= ANGLE_STRAIGHT_DEG) and (ang_dip >= ANGLE_STRAIGHT_DEG)

def finger_states(lm):
    # returns list [thumb, index, middle, ring, pinky] where 1=open/straight
    states = []
    for name in ORDER:
        states.append(1 if is_finger_straight(lm, FINGERS[name]) else 0)
    return states

# -------------------------
# GESTURE RULES
# -------------------------
def thumb_direction_up_down(lm, handedness_label):
    # Use vector MCP(2) -> TIP(4); y grows downward in image space
    tip = lm[4]; mcp = lm[2]
    dy = tip.y - mcp.y
    # compensate slight rotation by comparing magnitude vs palm width
    pw = palm_width(lm)
    up   = dy < -0.15 * pw
    down = dy >  0.15 * pw
    return up, down

def is_korean_love(lm):
    # thumb tip ↔ index tip close, other three fingers folded
    pw = palm_width(lm)
    thumb_tip = lm[4]; index_tip = lm[8]
    d = dist_norm(thumb_tip, index_tip)
    others_folded = (
        not is_finger_straight(lm, FINGERS["middle"]) and
        not is_finger_straight(lm, FINGERS["ring"]) and
        not is_finger_straight(lm, FINGERS["pinky"])
    )
    return (d < LOVE_DIST_FACTOR * pw) and others_folded

def classify_single_hand(lm, handedness_label):
    states = finger_states(lm)  # [thumb, index, middle, ring, pinky]
    s = sum(states)
    thumb_up, thumb_down = thumb_direction_up_down(lm, handedness_label)

    # Single-hand gestures
    # Thumbs Up/Down = thumb straight + 4 others folded + direction
    if states[0] == 1 and states[1] == 0 and states[2] == 0 and states[3] == 0 and states[4] == 0:
        if thumb_up:
            return "Good", states
        if thumb_down:
            return "Bad", states

    # Point = only index straight
    if states[0] == 0 and states[1] == 1 and states[2] == 0 and states[3] == 0 and states[4] == 0:
        return "Point", states

    # Run = fist (no fingers straight)
    if s == 0:
        return "Run", states

    # Korean Love by pose (thumb/index tips close, middle/ring/pinky folded)
    if is_korean_love(lm):
        return "Korean Love", states

    # Open hand (5 straight) — used along with other hand for Stop
    if s == 5:
        return "Open", states

    return "Unknown", states

def classify_two_hands(per_hand):
    """
    per_hand: list of tuples (label, states) for detected hands (<=2).
    We’ll merge into final labels:
      - Stop: both hands Open (total 10 fingers)
      - Otherwise prefer any decisive single-hand gesture if present
    """
    labels = [lab for lab, _ in per_hand]
    sums = [sum(st) for _, st in per_hand]

    # If two hands, check Stop
    if len(per_hand) == 2:
        if labels[0] == "Open" and labels[1] == "Open" and (sums[0] == 5 and sums[1] == 5):
            return "Stop", 10

    # Priority order for single-hand gestures if not Stop
    priority = ["Korean Love", "Good", "Bad", "Point", "Run", "Open"]
    for p in priority:
        if p in labels:
            return p, sum(sums)

    # If nothing strong detected
    total = sum(sums) if sums else 0
    return "Unknown", total

# -------------------------
# STABILITY / DEBOUNCE
# -------------------------
prev_label = "Unknown"
stable_label = "Unknown"
count_same = 0

def update_stable(label_now):
    global prev_label, stable_label, count_same
    if label_now == prev_label:
        count_same += 1
    else:
        prev_label = label_now
        count_same = 1
    if count_same >= STABLE_FRAMES:
        stable_label = label_now
    return stable_label

# -------------------------
# MAIN LOOP
# -------------------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    per_hand = []
    total_fingers = 0

    if res.multi_hand_landmarks:
        # pair landmarks with handedness label (Left/Right)
        handed_labels = []
        if res.multi_handedness:
            for hnd in res.multi_handedness:
                handed_labels.append(hnd.classification[0].label)  # "Left" or "Right"

        for i, lmset in enumerate(res.multi_hand_landmarks):
            label_lr = handed_labels[i] if i < len(handed_labels) else "Unknown"
            mp_draw.draw_landmarks(
                frame, lmset, mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )
            lbl, states = classify_single_hand(lmset.landmark, label_lr)
            per_hand.append((lbl, states))
            total_fingers += sum(states)

    final_label, tf = classify_two_hands(per_hand)
    final_label = update_stable(final_label)

    # HUD
    cv2.putText(frame, f"Gesture: {final_label}", (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.putText(frame, f"Fingers: {tf}", (40, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
    cv2.putText(frame, "ESC to quit", (40, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
