CLASS_LIST = [
    "ceramic",
    "fabric",
    "ground",
    "leather",
    "metal",
    "stone",
    "wood",
]

CLASS_PALETTE = {
    0: (255, 198, 138),  # ceramic, Pale Orange
    1: (216, 27, 96),  # fabric, Raspberry
    2: (139, 195, 74),  # ground, Olive Green
    3: (141, 110, 99),  # leather, Saddle Brown
    4: (0, 145, 233),  # metal, Blue
    5: (120, 144, 156),  # stone, Slate Gray
    6: (229, 115, 115),  # wood, Burnt Sienna
}

CLASS_LIST_IDX_MAPPING = {name: idx for idx, name in enumerate(CLASS_LIST)}
