import re
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

_TIMESTAMP_PATTERN = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3})')


class ClinicalGroup(Enum):
    CHR = 'CHR'
    HC = 'HC'
    UNKNOWN = 'UNKNOWN'


class TranscriptLine(NamedTuple):
    line_index: int
    speaker: str
    timestamp: str
    text: str


class Transcript:
    directory_path: Path | None = None

    @classmethod
    def set_directory_path(cls, path: str | Path) -> None:
        cls.directory_path = Path(path)

    def __init__(self, filename: str):
        if Transcript.directory_path is None:
            raise ValueError("directory_path is not set, call Transcript.set_directory_path() first and point towards the transcript directory.")
        self.filename: str = filename
        self.full_path: Path = Transcript.directory_path / filename
        self.group_status: ClinicalGroup = self.get_clinical_status()
        self.lines = self.get_text()

    @cached_property
    def interviewer_lines(self) -> list[TranscriptLine]:
        return [line for line in self.lines if line.speaker == "INTERVIEWER"]

    @cached_property
    def participant_lines(self) -> list[TranscriptLine]:
        return [line for line in self.lines if line.speaker == "PARTICIPANT"]

    def get_clinical_status(self) -> ClinicalGroup:
        for parent in self.full_path.parents:
            if parent.name.upper() == 'CHR':
                return ClinicalGroup.CHR
            elif parent.name.upper() == 'HC':
                return ClinicalGroup.HC
        print(f"ClinicalGroup not found for {self.full_path}, defaulting to UNKNOWN")
        return ClinicalGroup.UNKNOWN

    def get_text(self) -> list[TranscriptLine]:
        with open(self.full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        index: int = 0
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line == "":
                continue
            index += 1
            if line.startswith('PARTICIPANT'):
                speaker = "PARTICIPANT"
            else:
                speaker = "INTERVIEWER"
            match = _TIMESTAMP_PATTERN.search(line)
            if match:
                timestamp = match.group(1)
                text = line[match.end():].strip()
            else:
                timestamp = ""
                text = line.split(':', 1)[-1].strip()
            lines.append(TranscriptLine(index, speaker, timestamp, text))
        return lines

"""

from transcript import Transcript, ClinicalGroup

Transcript.set_directory_path("/data//transcripts")

t1 = Transcript("CHR/subject_001/interview.txt")
t2 = Transcript("HC/subject_042/interview.txt")

print(t1.group_status)           # ClinicalGroup.CHR
print(len(t1.lines))             # Total line count
print(len(t1.participant_lines)) # Participant utterances only

for line in t1.participant_lines:
    print(f"{line.line_number}: {line.timestamp} - {line.text}")

"""