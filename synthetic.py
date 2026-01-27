import random
from typing import Dict, List

from .data import PatientTrajectory


def generate_synthetic(num_patients: int = 200, vocab_size: int = 200) -> List[PatientTrajectory]:
    modalities = ["dx", "proc", "med", "lab"]
    roots = {m: list(range(i * vocab_size, (i + 1) * vocab_size)) for i, m in enumerate(modalities)}
    trajectories: List[PatientTrajectory] = []
    for pid in range(num_patients):
        num_visits = random.randint(3, 8)
        visits = []
        state = {m: random.choice(roots[m]) for m in modalities}
        for _ in range(num_visits):
            visit: Dict[str, List[int]] = {}
            for m in modalities:
                codes = {state[m], (state[m] + 3) % (vocab_size * 4)}
                visit[m] = list(codes)
                # small progression
                if random.random() > 0.6:
                    state[m] = (state[m] + random.randint(1, 5)) % (vocab_size * 4)
            visits.append(visit)
        trajectories.append(PatientTrajectory(subject_id=str(pid), visits=visits))
    return trajectories
