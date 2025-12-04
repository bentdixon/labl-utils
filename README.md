## Instructions

``` bash
cd your-project
git clone https://github.com/bentdixon/labl-utils.git
```

### Using `transcript_utils`

``` python
from utils.transcript_utils import Transcript, ClinicalGroup

# Must set the transcript directory first, containing CHR and HC subdirectories
Transcript.set_directory_path("/data/transcripts")

# Iterate through all transcripts
for transcript in Transcript.list_transcripts():
    print(transcript.filename)
    print(len(transcript.participant_lines))

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

### Using `combine_data`

``` python
from utils.combine_data import combine_csvs
import polars as pl

# Expects CSV with `patient_id` column, optionally `clinical_status`, and then metric columns
c1 = "some/path/data.csv" # `patient_id`, `metric_1`
c2 = "some/path/data.csv" # `patient_id`, `clinical_status`, `_metric_2`

# Results in combined Polars DataFrame with `patient_id`, `clinincal_status`, `metric_1`, `metric_2`, matched on `patient_id`
combined_data = combine_csvs(c1, c2)
```