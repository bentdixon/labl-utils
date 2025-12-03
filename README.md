## Instructions
```bash
cd your-project
git clone https://github.com/bentdixon/labl-utils/tree/main
```

### Using TranscriptUtils
```python
from utils.transcript import Transcript, ClinicalGroup

# Must set the transcript directory first, containing CHR and HC subdirectories
Transcript.set_directory_path("/data/transcripts")

# Define transcript objects
t1 = Transcript("CHR/subject_001/interview.txt")
t2 = Transcript("HC/subject_042/interview.txt")

# Access properties
print(t1.group_status)           # ClinicalGroup.CHR
print(len(t1.lines))             # Total line count
print(len(t1.participant_lines)) # Participant utterances only
print(len(t1.interviewer_lines)) # Interviewer utterances only

# Iterate over lines
for line in t1.participant_lines:
    print(f"{line.line_number}: {line.timestamp} - {line.text}")
```