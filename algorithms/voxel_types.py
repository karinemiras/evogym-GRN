VOXEL_TYPES = {
    'bone': 1,
    'fat': 2,
    'muscle_h': 3,
    'muscle_v': 4,
}
#  matches materials order in prepare_robot_files.py

TF_WEIGHTS = {
    'bone': 4,
    'fat': 4,
    'muscle_h': 3.0,
    'muscle_v': 3.0,
    'regulatory': 0.4,   # applies per regulatory TF
}

VOXEL_TYPES_COLORS = {
    'bone': (38, 38, 38), # EvoGym rigid/fixed dark gray
    'fat': (191, 191, 191), # EvoGym soft light gray
    'muscle_h': (230, 115, 30), # EvoGym horizontal actuator orange
    'muscle_v': (67, 161, 209), # EvoGym vertical actuator blue
}
