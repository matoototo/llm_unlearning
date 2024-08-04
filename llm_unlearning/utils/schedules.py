import math
from typing import Callable, Dict, Any

def cosine_annealing(start_factor: float = 1.0, end_factor: float = 0.0, time_scale: float = 1.0) -> Callable[[float], float]:
    def schedule(t: float) -> float:
        if t < time_scale:
            cosine_decay = (1 + math.cos(math.pi * t / time_scale)) / 2
            return end_factor + (start_factor - end_factor) * cosine_decay
        else:
            return end_factor
    return schedule

def linear_decay(start_factor: float = 1.0, end_factor: float = 0.0, time_scale: float = 1.0) -> Callable[[float], float]:
    def schedule(t: float) -> float:
        if t < time_scale:
            return start_factor + (end_factor - start_factor) * (t / time_scale)
        else:
            return end_factor
    return schedule

def get_schedule(config: Dict[str, Any]) -> Callable[[float], float]:
    schedules = {
        "cosine": cosine_annealing,
        "linear": linear_decay,
    }

    schedule_name = config.pop("name")
    if schedule_name not in schedules:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    schedule_func = schedules[schedule_name]

    try:
        return schedule_func(**config)
    except TypeError as e:
        raise ValueError(f"Invalid parameters for schedule '{schedule_name}': {str(e)}")
