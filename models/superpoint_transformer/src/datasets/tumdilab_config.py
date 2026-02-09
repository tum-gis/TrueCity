import numpy as np

# =========================
# Classes
# =========================
CLASS_NAMES = [
    'RoadSurface',           # 0
    'GroundSurface',         # 1
    'RoadInstallations',     # 2
    'Vehicle',               # 3
    'Pedestrian',            # 4
    'WallSurface',           # 5
    'RoofSurface',           # 6
    'Door',                  # 7
    'Window',                # 8
    'BuildingInstallation',  # 9
    'Tree',                  # 10
    'Noise',                 # 11
    'ignored'                # 12
]

TUMDILAB_NUM_CLASSES = 12

CLASS_COLORS = np.asarray([
    [128, 128, 128],  # RoadSurface
    [210, 180, 140],  # GroundSurface
    [ 70, 130, 180],  # RoadInstallations
    [220,  20,  60],  # Vehicle
    [255, 215,   0],  # Pedestrian
    [199,  21, 133],  # WallSurface
    [100, 149, 237],  # RoofSurface
    [255, 127,  80],  # Door
    [135, 206, 250],  # Window
    [154, 205,  50],  # BuildingInstallation
    [ 34, 139,  34],  # Tree
    [105, 105, 105],  # Noise
    [  0,   0,   0],  # ignored
], dtype=np.float32)

# =========================
# (Optional) Instance/Panoptic
# =========================
STUFF_CLASSES = []

MIN_OBJECT_SIZE = 50
THING_CLASSES = [i for i in range(TUMDILAB_NUM_CLASSES) if i not in STUFF_CLASSES]

# =========================
# Splits
# =========================
SCANS = {'train': [], 'val': [], 'test': []}
