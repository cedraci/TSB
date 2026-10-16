"""Spell Brigade bot: best-rarity picks on every choice screen + ZQSD movement.

Run:   py autoclicker.py [--debug] [--no-move]
Keys:  F8 = pause/resume everything, F9 = toggle movement, Ctrl+C = quit.

- Choice screens (level-up "AMÉLIORATION", "AUGMENTATION", events, ...) are
  detected by the "SÉLECTIONNER" button hint in the bottom-right corner. The
  bot waits for the cards to finish animating, clicks the rarest card
  (red légendaire > purple épique > blue rare > white commun; ties are
  broken at random), and if that slot was empty it tries another one.
  "Gain de mana" and "Chance" upgrades (recognized by their card icon) are
  taken first whatever their rarity.
  Infusion screens (spell level 7 / 14) pick by element instead: acide >
  glace > foudre > feu > ténèbres.
  Works with the game in the background (it grabs focus only for the
  click, then gives it back).
- Movement (ZQSD, by physical key position so it works on AZERTY) only
  happens while the game is the foreground window, so keys never leak into
  other apps. It kites: always moving in wide arcs (bending toward XP
  orbs), and turning hard away when HP drops. When the camera stops moving
  although keys are held (wall, crowd) it steps aside or turns back. During mushroom events it
  walks onto every visible mushroom and roams to find more; during golden
  crab and mana-beast events it walks to the nearest green border marker;
  during sacrifice events it circles inside the sacrifice zone; during
  statue events it roams in long runs until the statues get markers; during
  rift events it goes to each rift and loops inside it until it closes.
"""
import ctypes
import ctypes.wintypes as w
import math
import os
import random
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

GAME_TITLE = "TheSpellBrigade"
HERE = os.path.dirname(os.path.abspath(__file__))
HINT_TEMPLATE = os.path.join(HERE, "hint_template.png")
HINT_BOX = (842, 497, 950, 517)       # "SÉLECTIONNER" hint, in 960x540 space
CARDS_BOX = (180, 120, 790, 420)      # where the cards live, in 960x540 space
CARD_X = (277 / 960, 480 / 960, 682 / 960)  # card centers, fraction of width
CARD_Y = 265 / 540
MATCH_THRESHOLD = 20                  # mean abs diff: ~1 on screen, ~85 off
SETTLED_DIFF = 4                      # frame diff below this = animation done
MIN_SCREEN_TIME = 1.5                 # cards fly in one by one, with pauses
STABLE_READS = 3                      # consecutive still frames (0.3s apart)
POLL = 0.4
DEBUG = "--debug" in sys.argv
DEBUG_DIR = os.path.join(HERE, "debug")
DEBUG_KEEP = 200                      # newest screenshots kept in debug/
VK_F8, VK_F9 = 0x77, 0x78

# Movement tuning (full-resolution pixels, 1920x1080 reference).
# Enemies look too much like the terrain to be seen reliably, so the bot
# kites: it never stands still, curving in wide arcs so the pack trails
# behind it, and turns hard away whenever it takes a hit.
KITE_TURN = math.radians(30)          # arc curvature, rad/s
KITE_FLIP = (5.0, 10.0)               # seconds between arc direction changes
ORB_STEER = math.radians(150)         # rad/s turning toward visible XP
FLEE_FOR = 1.0                        # straight run after a hit
ORB_RADIUS = 900                      # look for XP orbs within this distance
# Stuck detection: keys held but the camera (which follows the player) has not
# moved = blocked by a wall or a crowd. Then try +-90 deg, then reverse.
STUCK_AFTER = 0.45                    # seconds without camera motion
DETOUR_FOR = 0.8                      # commit to the new direction this long
CAM_W, CAM_H, CAM_R = 160, 90, 8      # camera tracking image size / search px
# Mana-beast event ("Rassemblez les bêtes de mana", banner title matched):
# walk to the nearest cow marker, touch the cow, then lead it to the enclosure
# in short steps with pauses so a following cow keeps up. The enclosure's
# glowing fence panels (long thin blue shapes) are remembered in map
# coordinates, tracked by adding up the camera shifts.
EVENT_TITLES = {                      # banner title templates (960x540 space)
    "cow": os.path.join(HERE, "cow_title_template.png"),
    "sacrifice": os.path.join(HERE, "sacrifice_title_template.png"),
    "statues": os.path.join(HERE, "statues_title_template.png"),
    "rift": os.path.join(HERE, "rift_title_template.png"),
}
EVENT_TITLE_BOX = (380, 30, 580, 47)
EVENT_TITLE_MATCH = 25                # ~0-16 during the event, >45 otherwise
ESCORT_FOR = 30.0                     # give up leading a cow after this
ESCORT_STEP = (0.35, 0.25)            # seconds walking / waiting
PEN_REACHED = 60                      # px from the enclosure center
# Sacrifice event ("Terminez le sacrifice"): monsters must die inside the
# orange-red circle around the shrine. The bot circles inside it so the
# monsters chasing it are dragged in; it follows the green marker when the
# circle is off-screen.
ALTAR_ORBIT = 60                      # px from the circle's center
# Statues event ("Écrasez les statues"): statues get green markers only in the
# last minute. Before that the bot roams in long, nearly straight runs (no XP
# detours) to pass by as many statues as possible.
ROAM_TURN = math.radians(8)           # rad/s, vs KITE_TURN when kiting
# Rift event ("Scellez la Faille"): 1-3 circles to stand in until they close.
# The green marker points at one of them (and sits on it when on screen, where
# it is masked out near the player). The bot remembers where the marker
# points on the map and loops tightly there until a marker shows elsewhere.
RIFT_ORBIT = 25                       # px from the rift center: stay inside
RIFT_NEW = 150                        # marker this far away = another rift
# Events ("Récupérez les champignons ensorcelés"): the mushrooms are solid
# light-blue blobs with a cyan outline. While one is visible the bot walks
# straight at it; after that it keeps exploring for HUNT_FOR seconds.
HUNT_FOR = 25.0
MUSHROOM_MIN_PX = 100                 # partly hidden mushrooms are skipped
CHASE_FOR = 0.4                       # re-aim at the mushroom this often
# Events ("Traquez le Crabe doré"): a pure-green skull (~0,243,101) on the
# screen border points at the golden crab. The bot walks toward it, and keeps
# that heading for CRAB_PURSUE seconds once the crab is on screen and the
# skull disappears, so the spells (auto-aimed) can finish it.
MARKER_MIN_PX = 60                    # skull ~280 px, cow head ~200; digits < 20
CRAB_PURSUE = 4.0
SCAN = {"up": 0x11, "left": 0x1E, "down": 0x1F, "right": 0x20}  # Z Q S D on AZERTY

u = ctypes.windll.user32
g = ctypes.windll.gdi32
u.SetProcessDPIAware()


class BIH(ctypes.Structure):
    _fields_ = [("biSize", w.DWORD), ("biWidth", w.LONG), ("biHeight", w.LONG),
                ("biPlanes", w.WORD), ("biBitCount", w.WORD), ("biCompression", w.DWORD),
                ("biSizeImage", w.DWORD), ("biXPelsPerMeter", w.LONG), ("biYPelsPerMeter", w.LONG),
                ("biClrUsed", w.DWORD), ("biClrImportant", w.DWORD)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def client_rect(h):
    r = w.RECT()
    u.GetClientRect(h, ctypes.byref(r))
    p = w.POINT(0, 0)
    u.ClientToScreen(h, ctypes.byref(p))
    return p.x, p.y, r.right, r.bottom


def grab(h):
    """Capture the game's client area even if covered (PrintWindow, full content)."""
    _, _, W, H = client_rect(h)
    if W <= 0 or H <= 0:
        return None
    hdc = u.GetDC(h)
    mdc = g.CreateCompatibleDC(hdc)
    bmp = g.CreateCompatibleBitmap(hdc, W, H)
    g.SelectObject(mdc, bmp)
    u.PrintWindow(h, mdc, 3)
    bi = BIH(ctypes.sizeof(BIH), W, -H, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(W * H * 4)
    g.GetDIBits(mdc, bmp, 0, H, buf, ctypes.byref(bi), 0)
    g.DeleteObject(bmp)
    g.DeleteDC(mdc)
    u.ReleaseDC(h, hdc)
    return Image.frombuffer("RGB", (W, H), buf, "raw", "BGRX", 0, 1)


def gray(small, box):
    return np.asarray(small.crop(box).convert("L"), dtype=np.float32)


def save_debug(small, tag):
    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        small.save(os.path.join(DEBUG_DIR, f"{time.strftime('%H%M%S')}_{tag}.png"))
        files = sorted(os.listdir(DEBUG_DIR), key=lambda f: os.path.getmtime(os.path.join(DEBUG_DIR, f)))
        for f in files[:-DEBUG_KEEP]:     # never fill the disk (3000+ files once did)
            os.remove(os.path.join(DEBUG_DIR, f))


# ---------------------------------------------------------------- input ----

def focus(h):
    if u.GetForegroundWindow() == h:
        return
    u.keybd_event(0x12, 0, 0, 0)      # tapping Alt lets us take the foreground
    u.keybd_event(0x12, 0, 2, 0)
    u.SetForegroundWindow(h)
    time.sleep(0.15)


def click(x, y):
    u.SetCursorPos(x, y)
    time.sleep(0.12)                  # let the game register the hover
    u.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.05)
    u.mouse_event(0x0004, 0, 0, 0, 0)


def key(scan, down):
    u.keybd_event(0, scan, 0x8 | (0 if down else 0x2), 0)  # KEYEVENTF_SCANCODE


class Mover:
    def __init__(self):
        self.held = set()
        self.release_at = 0.0
        self.last_hp = None
        self.suspect = False
        self.hunt_until = 0.0
        self.crab_until = 0.0
        self.rec_at = 0.0
        self.heading = random.uniform(0, 2 * math.pi)
        self.turn = random.choice((-1, 1))
        self.flip_at = time.time() + random.uniform(*KITE_FLIP)
        self.flee_until = 0.0
        self.last_tick = time.time()
        self.prev_cam = None
        self.still_since = None
        self.stuck_tries = 0
        self.detour_until = 0.0
        self.detour_heading = 0.0
        self.world = np.zeros(2)          # player position on the map (960x540 px)
        self.pen = None                   # enclosure position on the map
        self.cow_event_until = 0.0
        self.sac_event_until = 0.0
        self.roam_until = 0.0
        self.rift_until = 0.0
        self.rift = None                  # current rift position on the map
        self.altar = None                 # sacrifice circle center on the map
        self.cow_near_until = 0.0
        self.escort_until = 0.0

    def release(self):
        for k in self.held:
            key(SCAN[k], False)
        self.held = set()

    def go(self, dx, dy, duration):
        """Hold the keys for direction (dx, dy) in screen space (y down)."""
        ang = math.atan2(dy, dx)
        want = set()
        if math.cos(ang) > 0.38:
            want.add("right")
        if math.cos(ang) < -0.38:
            want.add("left")
        if math.sin(ang) > 0.38:
            want.add("down")
        if math.sin(ang) < -0.38:
            want.add("up")
        for k in self.held - want:
            key(SCAN[k], False)
        for k in want - self.held:
            key(SCAN[k], True)
        self.held = want
        self.release_at = time.time() + duration

    def tick(self, img, game_focused):
        now = time.time()
        dt = min(now - self.last_tick, 0.5)
        self.last_tick = now
        if self.held and (now >= self.release_at or not game_focused):
            self.release()
        if not game_focused:
            self.last_hp = None
            return
        a = np.asarray(img.resize((960, 540)), dtype=np.int16)
        self.check_stuck(img, now, dt)
        hp = hp_ratio(a)
        hurt = False
        if hp is not None and hp > 0.05:  # bar hidden (fades, overlays) reads ~0
            dropped = self.last_hp is not None and hp < self.last_hp - 0.01
            hurt = dropped and self.suspect   # must persist across two reads
            self.suspect = dropped and not hurt
            if not self.suspect:
                self.last_hp = hp
        if now < self.detour_until:       # stepping around a wall / crowd
            self.heading = self.detour_heading
            self.go(math.cos(self.heading), math.sin(self.heading), 0.6)
            return
        orbs, mushrooms = scan_field(a)
        marker = event_marker(a)
        kind = event_kind(a)
        if kind == "cow":
            self.cow_event_until = now + 5.0
        elif kind == "sacrifice":
            self.sac_event_until = now + 5.0
        elif kind == "rift":
            self.rift_until = now + 5.0
        elif kind == "statues":
            if now >= self.roam_until:
                log("statues event -> roaming wide to find them")
            self.roam_until = now + 5.0
        if now < self.rift_until:
            if marker:
                seen = self.world + np.array(marker, dtype=float)
                if self.rift is None or math.hypot(*(seen - self.rift)) > RIFT_NEW:
                    log(f"rift marker at ({marker[0]:+d},{marker[1]:+d}) -> going to close it")
                    self.rift = seen
                else:                         # refine while approaching
                    self.rift = 0.7 * self.rift + 0.3 * seen
            if self.rift is not None:
                self.circle_altar(now, dt, self.rift, RIFT_ORBIT)
                return
        else:
            self.rift = None
        if now < self.sac_event_until:
            ring = ring_center(a)
            if ring is not None:
                seen = self.world + ring
                if self.altar is None:
                    log(f"sacrifice circle at ({ring[0]:+.0f},{ring[1]:+.0f}) -> circling inside it")
                self.altar = seen if self.altar is None else 0.7 * self.altar + 0.3 * seen
            if self.altar is not None:
                self.circle_altar(now, dt, self.altar, ALTAR_ORBIT)
                return
        else:
            self.altar = None
        if now < self.cow_event_until:
            fence = fence_center(a)
            if fence is not None:
                seen = self.world + fence
                self.pen = seen if self.pen is None else 0.8 * self.pen + 0.2 * seen
            if self.herd(now, marker):
                return
        else:
            self.pen, self.escort_until = None, 0.0
        if marker:                        # event: go for its target
            x, y = marker
            if now >= self.crab_until:
                log(f"event marker at ({x:+d},{y:+d}) -> going for it")
            self.crab_until = now + CRAB_PURSUE
            if DEBUG and now >= self.rec_at:  # footage for tuning event logic
                self.rec_at = now + 2.0
                save_debug(img.resize((960, 540)), "event")
            self.heading = math.atan2(y, x)
            self.go(x, y, CHASE_FOR)
            return
        if now < self.crab_until:         # crab just came on screen: close in
            self.go(math.cos(self.heading), math.sin(self.heading), CHASE_FOR)
            return
        if mushrooms:                     # event: walk onto the nearest mushroom
            x, y = min(mushrooms, key=lambda p: math.hypot(*p))
            if not self.hunting(now):
                log(f"mushroom spotted at ({x:+d},{y:+d}) -> collecting")
            self.hunt_until = now + HUNT_FOR
            self.heading = math.atan2(y, x)
            self.go(x, y, CHASE_FOR)
            return
        if hurt and now >= self.flee_until:   # something is ahead or on us: turn away
            self.heading += math.pi + random.uniform(-math.pi / 3, math.pi / 3)
            self.turn = -self.turn
            self.flee_until = now + FLEE_FOR
            log(f"hit (HP {hp:.0%}) -> turning away")
        elif now >= self.flee_until:
            if now >= self.flip_at:
                self.turn = -self.turn
                self.flip_at = now + random.uniform(*KITE_FLIP)
            roaming = now < self.roam_until
            target = None if self.hunting(now) or roaming else orb_direction(orbs)
            if target is None:
                self.heading += self.turn * (ROAM_TURN if roaming else KITE_TURN) * dt
            else:                         # bend the arc toward XP
                diff = (target - self.heading + math.pi) % (2 * math.pi) - math.pi
                self.heading += max(-ORB_STEER * dt, min(ORB_STEER * dt, diff))
        self.go(math.cos(self.heading), math.sin(self.heading), 0.6)

    def herd(self, now, marker):
        """Mana-beast event step. True when it decided the movement."""
        if marker:                        # a cow off-screen (or one we lost)
            x, y = marker
            if now < self.escort_until:
                log("cow fell behind -> going back for it")
                self.escort_until = 0.0
            self.cow_near_until = now + CRAB_PURSUE
            self.heading = math.atan2(y, x)
            self.go(x, y, CHASE_FOR)
            return True
        if now < self.cow_near_until:     # cow just came on screen: touch it
            self.go(math.cos(self.heading), math.sin(self.heading), CHASE_FOR)
            if now + 0.15 >= self.cow_near_until and self.pen is not None:
                self.escort_until = now + ESCORT_FOR
                log("reached a cow -> leading it to the enclosure")
            return True
        if now < self.escort_until and self.pen is not None:
            to = self.pen - self.world
            if math.hypot(*to) < PEN_REACHED:
                log("cow brought to the enclosure")
                self.escort_until = 0.0
                return False
            walk, wait = ESCORT_STEP
            if (now % (walk + wait)) < walk:  # slow pace: a follower keeps up
                self.heading = math.atan2(to[1], to[0])
                self.go(to[0], to[1], walk)
            else:
                self.release()
            return True
        return False

    def circle_altar(self, now, dt, center, orbit):
        """Keep moving in a small loop of radius ~orbit around a map point."""
        to = center - self.world
        d = math.hypot(*to)
        if now >= self.flip_at:
            self.turn = -self.turn
            self.flip_at = now + random.uniform(*KITE_FLIP)
        # far: head in; at ALTAR_ORBIT: tangent; never further out than tangent
        self.heading = math.atan2(to[1], to[0]) + self.turn * (math.pi / 2) * min(1.0, orbit / max(d, 1))
        self.go(math.cos(self.heading), math.sin(self.heading), 0.6)

    def check_stuck(self, img, now, dt):
        """Detect 'keys held, camera not moving' and schedule a detour."""
        cam = np.asarray(img.convert("L").resize((CAM_W, CAM_H), Image.BILINEAR), dtype=np.float32)
        prev, self.prev_cam = self.prev_cam, cam
        if prev is None or dt >= 0.5:     # a gap (card screen, unfocused)
            self.still_since = None
            return
        dx, dy, live = camera_shift(prev, cam)
        if not live:                      # frozen frame: not gameplay
            return
        self.world -= np.array([dx, dy]) * (960 / CAM_W)   # scenery moves opposite
        if not self.held:                 # standing still on purpose
            return
        if dx or dy:
            self.still_since, self.stuck_tries = None, 0
            return
        if self.still_since is None:
            self.still_since = now
        elif now - self.still_since >= STUCK_AFTER:
            self.stuck_tries += 1
            if self.stuck_tries == 1:     # turns are relative to the blocked heading
                self.side = random.choice((-1, 1))
                self.stuck_base = self.heading
            turn = (self.side * math.pi / 2, -self.side * math.pi / 2, math.pi)[(self.stuck_tries - 1) % 3]
            self.detour_heading = self.stuck_base + turn
            self.detour_until = now + DETOUR_FOR
            self.still_since = None
            log(f"stuck -> detour {math.degrees(turn):+.0f} deg (try {self.stuck_tries})")

    def hunting(self, now):
        return now < self.hunt_until


def event_marker(a):
    """Offset from the player of the nearest pure-green event marker, or None.

    Events pin a green icon on the screen border toward their target: a skull
    for the golden crab, cow heads for the mana beasts (often several, along
    the bottom edge)."""
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    m = (G > 170) & (R < 70) & (G > B + 60)
    m[440:505, 380:580] = False       # spell icons (acid one is this green)
    m[200:325, 400:560] = False       # green heal digits around the player
    lab, _ = ndimage.label(m)
    best = None
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if (lab[sl] == i).sum() < MARKER_MIN_PX:
            continue
        ys, xs = np.nonzero(lab[sl] == i)
        x, y = int(xs.mean()) + sl[1].start - 480, int(ys.mean()) + sl[0].start - 262
        if best is None or math.hypot(x, y) < math.hypot(*best):
            best = (x, y)
    return best


EVENT_TITLE_IMGS = {k: np.asarray(Image.open(v).convert("L"), dtype=np.float32)
                    for k, v in EVENT_TITLES.items()}


def event_kind(a):
    """Name of the event whose banner is on screen ("cow", "sacrifice"), or None."""
    x0, y0, x1, y1 = EVENT_TITLE_BOX
    g_ = a[y0:y1, x0:x1].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    for k, t in EVENT_TITLE_IMGS.items():
        if np.abs(g_ - t).mean() < EVENT_TITLE_MATCH:
            return k
    return None


def ring_center(a):
    """Screen offset of the sacrifice circle's center, fitted (RANSAC) on its
    orange-red rim so a partly visible circle works too, or None."""
    R, G, B = a[..., 0].astype(np.int16), a[..., 1].astype(np.int16), a[..., 2].astype(np.int16)
    m = (R > 170) & (G > 40) & (G < 130) & (B < 90) & (R > G + 80)
    m[:50] = False
    m[440:] = False
    ys, xs = np.nonzero(m[::2, ::2])
    if len(xs) < 150:
        return None
    xs, ys = xs * 2.0, ys * 2.0
    rng = np.random.default_rng()
    best = None
    for _ in range(150):              # circle through 3 random rim points
        i = rng.choice(len(xs), 3, replace=False)
        x1, x2, x3 = xs[i]
        y1, y2, y3 = ys[i]
        A_ = np.array([[x2 - x1, y2 - y1], [x3 - x1, y3 - y1]]) * 2
        if abs(np.linalg.det(A_)) < 1e-3:
            continue
        cx, cy = np.linalg.solve(A_, [x2**2 - x1**2 + y2**2 - y1**2, x3**2 - x1**2 + y3**2 - y1**2])
        r = math.hypot(x1 - cx, y1 - cy)
        if not 80 < r < 400:
            continue
        n = int((np.abs(np.hypot(xs - cx, ys - cy) - r) < 6).sum())
        if best is None or n > best[2]:
            best = (cx, cy, n)
    if best is None or best[2] < 300:     # real rims: 500-850 inliers
        return None
    return np.array([best[0] - 480, best[1] - 262])


def fence_center(a):
    """Screen offset of the enclosure, from its glowing lavender fence panels
    (long thin lines), or None when not enough fence is visible."""
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    lav = (B > 140) & (B > G + 25) & (B > R + 20) & (R > 70)
    lav[:50] = False
    lav[440:] = False
    lab, _ = ndimage.label(lav)
    xs, ys, ws = [], [], []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = int((lab[sl] == i).sum())
        if area >= 30 and max(w, h) >= 30 and max(w, h) >= 3 * min(w, h):
            xs.append(sl[1].start + w / 2)
            ys.append(sl[0].start + h / 2)
            ws.append(area)
    if sum(ws) < 120:
        return None
    return np.array([np.average(xs, weights=ws) - 480, np.average(ys, weights=ws) - 262])


CAM_MASK = np.zeros((CAM_H, CAM_W), bool)
CAM_MASK[8:72, 6:154] = True           # terrain: no HUD bars...
CAM_MASK[27:63, 53:107] = False        # ...and not the player's own effects


def camera_shift(a, b):
    """(dx, dy, live): camera motion between two CAM_W x CAM_H gray frames.

    live is False when the frames are identical (paused / frozen screen)."""
    R = CAM_R
    ca, m = a[R:CAM_H - R, R:CAM_W - R], CAM_MASK[R:CAM_H - R, R:CAM_W - R]
    best = (0, 0, float("inf"))
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            cb = b[R + dy:CAM_H - R + dy, R + dx:CAM_W - R + dx]
            e = np.minimum(np.abs(cb - ca), 40)[m].mean()
            if e < best[2]:
                best = (dx, dy, e)
    live = np.minimum(np.abs(b - a), 40)[CAM_MASK].mean() > 2
    return best[0], best[1], live


def orb_direction(orbs):
    """Angle toward the nearby XP orbs (closer orbs weigh more), or None."""
    vx, vy = 0.0, 0.0
    for (x, y) in orbs:
        d = math.hypot(x, y)
        if 0 < d < ORB_RADIUS / 2:
            vx += x / (d * (d + 75))
            vy += y / (d * (d + 75))
    return math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-6 else None


def hp_ratio(a):
    """Filled fraction of the orange HP bar (960x540 space)."""
    row = a[510, 336:625]
    filled = (row[:, 0] > 150) & (row[:, 2] < 80)
    return float(filled.mean()) if row.size else None


def scan_field(a):
    """Return (orb offsets, mushroom offsets) from the player, in 960x540 pixels."""
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    blue = (B > 170) & (B > R + 70) & (B > G + 20)
    blue[:40] = False                 # top HUD
    blue[440:] = False                # spells / HP / XP bars
    blue[:110, 900:] = False          # top-right buff icons
    cy, cx = 262, 480                 # player sits at the screen center
    lab, _ = ndimage.label(blue)
    mushrooms, solid = [], np.zeros_like(blue)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        px = lab[sl] == i
        area = int(px.sum())
        # mushrooms / blue hexagons: 160-430 px, ~round, >45% filled, deep
        # blue (B-R >= 117). Fence panels are long and thin, XP orb clusters
        # sparse or tiny, the shield bubble's edge whiter (B-R ~80).
        if (area >= MUSHROOM_MIN_PX and area / (w * h) >= 0.45
                and max(w, h) <= 1.6 * min(w, h)
                and (B[sl][px] - R[sl][px]).mean() >= 100):
            mushrooms.append((sl[1].start + w // 2 - cx, sl[0].start + h // 2 - cy))
            solid[sl] |= lab[sl] == i
    ys, xs = np.nonzero((blue & ~solid)[::2, ::2])
    return list(zip(xs * 2 - cx, ys * 2 - cy)), mushrooms


# -------------------------------------------------------------- choices ----

def wait_settled(h, hint):
    """Wait for the card animation to finish. None if the screen went away."""
    prev, still, start = None, 0, time.time()
    for _ in range(25):
        img = grab(h)
        if img is None:
            return None
        small = img.resize((960, 540))
        if np.abs(gray(small, HINT_BOX) - hint).mean() >= MATCH_THRESHOLD:
            return None
        cur = gray(small, CARDS_BOX)
        if prev is not None and np.abs(cur - prev).mean() < SETTLED_DIFF:
            still += 1
            if still >= STABLE_READS and time.time() - start >= MIN_SCREEN_TIME:
                return small, cur
        else:
            still = 0
        prev = cur
        time.sleep(0.3)
    return small, cur


RARITIES = ("commun", "rare", "épique", "légendaire")   # worst -> best
RARITY_MIN_PX = 60                    # colored label/glow: ~200-800 px, else 0
INFUSION_TEMPLATE = os.path.join(HERE, "infusion_template.png")
TITLE_BOX = (380, 48, 580, 82)        # screen title, in 960x540 space
LABEL_BAND = (330, 375)               # rows of the card's bottom label / glow
# Infusion element priority, best first. Measured label colors: acide
# (123,168,31) hue ~80, glace (202,246,247) pale cyan, feu (207,126,9) hue ~35.
# Ténèbres is purple; foudre not seen yet: guessed yellow or saturated blue.
ELEMENTS = ("acide", "glace", "foudre", "feu", "ténèbres")


def hue_of(rgb):
    """Hue in degrees (0-360) of float RGB arrays in [0, 1], last axis = RGB."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = np.where(mx - mn == 0, 1, mx - mn)
    return np.where(mx == R, (G - B) / d % 6,
                    np.where(mx == G, (B - R) / d + 2, (R - G) / d + 4)) * 60


def card_rarity(small, i):
    """0-3 rarity of card i from its label/glow color (white, blue, purple, red)."""
    cx = int(CARD_X[i] * 960)
    reg = np.asarray(small.crop((cx - 60, 315, cx + 60, 380)), dtype=np.float32) / 255
    mx, mn = reg.max(2), reg.min(2)
    vivid = (mx - mn > 0.35) & (mx > 0.5)
    hue = hue_of(reg)
    counts = {3: ((hue < 30) | (hue > 340)) & vivid,
              2: (hue > 260) & (hue < 320) & vivid,
              1: (hue > 170) & (hue < 230) & vivid}
    best = max(counts, key=lambda k: counts[k].sum())
    return best if counts[best].sum() >= RARITY_MIN_PX else 0


def element_of(rgb):
    """Element name from the mean color of an infusion card's label/glow."""
    r, g, b = rgb
    mx, mn = max(rgb), min(rgb)
    if mx - mn < 0.3 * mx and b > r:  # pale cyan
        return "glace"
    h = float(hue_of(np.array(rgb)))
    if h < 45 or h >= 330:
        return "feu"
    if h < 62 or 200 <= h < 250:
        return "foudre"
    if h < 110:
        return "acide"
    if 250 <= h < 330:
        return "ténèbres"
    return "?"


def infusion_cards(small):
    """Cards on an infusion screen: found by their colored labels (2 or 3)."""
    band = np.asarray(small, dtype=np.float32)[LABEL_BAND[0]:LABEL_BAND[1]] / 255
    mx, mn = band.max(2), band.min(2)
    vivid = ((mx - mn > 0.25) & (mx > 0.45)) | (mx > 0.85)
    runs = []
    for x in np.nonzero(vivid.sum(0) > 3)[0]:
        if runs and x - runs[-1][1] <= 12:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    cards = []
    for x0, x1 in runs:
        if x1 - x0 <= 40:
            continue
        m = vivid[:, x0:x1 + 1]
        rgb = band[:, x0:x1 + 1][m].mean(0)
        el = element_of(tuple(rgb))
        score = len(ELEMENTS) - ELEMENTS.index(el) if el in ELEMENTS else 0
        cards.append({"x": (x0 + x1) / 2 / 960, "score": score,
                      "label": el if el != "?" else f"?{tuple(int(v * 255) for v in rgb)}"})
    return cards


PRIORITY_ICONS = os.path.join(HERE, "priority_icons.npz")
PRIORITY_MATCH = 12                   # icon mean abs diff: <1 on match, >30 off
PRIORITY_BONUS = 10                   # beats any rarity
ICON_R = 16                           # icon templates are 32x32 (960x540 space)


def priority_icon(gray_small, cx, icons):
    """Name of the priority upgrade (gain de mana / chance) shown on the card, or None."""
    for name, t in icons.items():
        for cy in range(138, 172):        # hovered cards sit higher and bigger
            for x in range(cx - 7, cx + 8):
                win = gray_small[cy - ICON_R:cy + ICON_R, x - ICON_R:x + ICON_R]
                if np.abs(win - t).mean() < PRIORITY_MATCH:
                    return name.split("_")[0]
    return None


def read_cards(small, infusion, icons):
    """Cards on the current choice screen, each {x (fraction), score, label}."""
    if np.abs(gray(small, TITLE_BOX) - infusion).mean() < MATCH_THRESHOLD:
        cards = infusion_cards(small)
        if cards:
            return cards
    g_ = np.asarray(small.convert("L"), dtype=np.float32)
    cards = []
    for i in range(3):
        r = card_rarity(small, i)
        pr = priority_icon(g_, int(CARD_X[i] * 960), icons)
        cards.append({"x": CARD_X[i], "score": r + (PRIORITY_BONUS if pr else 0),
                      "label": RARITIES[r] + (f" +{pr}" if pr else "")})
    return cards


def handle_choices(h, hint, infusion, icons, mover, stats):
    mover.release()
    prev_fg, prev_pos = u.GetForegroundWindow(), w.POINT()
    u.GetCursorPos(ctypes.byref(prev_pos))
    dead = set()                      # card slots that did nothing on this screen
    for _ in range(10):
        res = wait_settled(h, hint)
        if res is None:
            break
        small, before = res
        cards = read_cards(small, infusion, icons)
        slots = [i for i in range(len(cards)) if i not in dead] or list(range(len(cards)))
        top = max(cards[i]["score"] for i in slots)
        choice = random.choice([i for i in slots if cards[i]["score"] == top])  # ties: random
        log("cards: " + ", ".join(f"{i + 1}={cards[i]['label']}" for i in slots))
        save_debug(small, f"before_card{choice + 1}")
        ox, oy, W, H = client_rect(h)
        focus(h)
        click(ox + int(cards[choice]["x"] * W), oy + int(CARD_Y * H))
        time.sleep(0.8)
        after = grab(h)
        if after is None:
            break
        after = after.resize((960, 540))
        if np.abs(gray(after, CARDS_BOX) - before).mean() < SETTLED_DIFF:
            dead.add(choice)          # empty slot / nothing happened: try another
            log(f"card {choice + 1} did nothing, retrying")
            continue
        stats["picks"] += 1
        dead = set()
        log(f"pick #{stats['picks']}: card {choice + 1}/{len(cards)} ({cards[choice]['label']})")
        save_debug(after, "after")
    if prev_fg and prev_fg != h:      # give focus/cursor back
        u.SetCursorPos(prev_pos.x, prev_pos.y)
        focus(prev_fg)


# ----------------------------------------------------------------- main ----

def main():
    hint = np.asarray(Image.open(HINT_TEMPLATE).convert("L"), dtype=np.float32)
    infusion = np.asarray(Image.open(INFUSION_TEMPLATE).convert("L"), dtype=np.float32)
    icons = dict(np.load(PRIORITY_ICONS))   # mana_big/small, chance_big/small
    move_on = "--no-move" not in sys.argv
    paused = False
    mover = Mover()
    stats = {"picks": 0}
    print("Spell Brigade bot running. F8 = pause/resume, F9 = toggle movement, Ctrl+C = quit.")
    print(f"Movement: {'on' if move_on else 'off'} (only while the game window is focused).")
    try:
        while True:
            if u.GetAsyncKeyState(VK_F8) & 1:
                paused = not paused
                mover.release()
                log("Paused." if paused else "Resumed.")
            if u.GetAsyncKeyState(VK_F9) & 1:
                move_on = not move_on
                mover.release()
                log(f"Movement {'on' if move_on else 'off'}.")
            h = u.FindWindowW(None, GAME_TITLE)
            if paused or not h or u.IsIconic(h):
                mover.release()
                time.sleep(POLL)
                continue
            img = grab(h)
            if img is None:
                time.sleep(POLL)
                continue
            small = img.resize((960, 540))
            if np.abs(gray(small, HINT_BOX) - hint).mean() < MATCH_THRESHOLD:
                handle_choices(h, hint, infusion, icons, mover, stats)
                continue
            if move_on:
                mover.tick(img, u.GetForegroundWindow() == h)
            busy = mover.held or time.time() < mover.escort_until
            time.sleep(0.1 if busy else POLL)
    finally:
        mover.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
