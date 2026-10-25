from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeVar, Generic, Optional

InputT = TypeVar("InputT")
PredictionT = TypeVar("PredictionT")
LabelT = TypeVar("LabelT")



@dataclass
class Observation(Generic[InputT, PredictionT, LabelT]):
    input: InputT
    prediction: PredictionT
    label: LabelT | None = None
    sample_index: Optional[int] = None
    time_stamp: datetime = field(default_factory=datetime.now)