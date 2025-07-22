CLASS_LIST = [
    "fabric",
    "ground",
    "leather",
    "metal",
    "stone",
    "wood",
    # "ceramic",
]

CLASS_PALETTE = {
    0: (216, 27, 96),  # fabric, Raspberry
    1: (139, 195, 74),  # ground, Olive Green
    2: (141, 110, 99),  # leather, Saddle Brown
    3: (0, 145, 233),  # metal, Blue
    4: (120, 144, 156),  # stone, Slate Gray
    5: (229, 115, 115),  # wood, Burnt Sienna
    6: (255, 198, 138),  # ceramic, Pale Orange -> ! Merged into stone
}

CLASS_LIST_IDX_MAPPING = {name: idx for idx, name in enumerate(CLASS_LIST)}
