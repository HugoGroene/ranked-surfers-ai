"""Single-use gameplay reading tied to one exact policy-source capture."""
from dataclasses import dataclass


@dataclass
class PreparedGameplayReading:
    image_identity: int
    captured_at: float
    state: str
    probabilities: dict
    menu_score: float

    def valid_for(self, image, captured_at, now, *, max_age_seconds=.020):
        age = float(now)-self.captured_at
        return (id(image) == self.image_identity and captured_at == self.captured_at
                and 0. <= age <= max_age_seconds)
