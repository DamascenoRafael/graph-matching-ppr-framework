from dataclasses import dataclass


@dataclass
class MatchingProgress:
    correct_matches: int
    matching_size: int
    max_possible_matches: int
    match_ratio: float

    @staticmethod
    def header() -> str:
        return "correct_matches,matching_size,max_possible_matches,match_ratio"

    def to_csv(self) -> str:
        return f"{self.correct_matches},{self.matching_size},{self.max_possible_matches},{self.match_ratio}"
