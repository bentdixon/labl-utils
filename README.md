# Instructions

``` bash
cd your-project
git clone https://github.com/bentdixon/labl-utils.git
cd labl-utils
# Make new virtual environment (Conda or Venv, etc.), or use pre-existing environment. If using pre-existing, use --no-deps. If creating new, use default.
pip install -e .  # OR, pip install -e . --no-deps
```

## Using `utils/`

Example usages of scripts and functions in the `utils/` subfolder.

### `transcript_utils.py`

Implements `Transcript` object - see Python file for complete list of parameters and methods. Allows for easy iteration through directories, easy text parsing, and automatic assignment of site location, clinical status (when available), and patient ID.

``` python
from utils.transcripts import Transcript

# Set the transcript directory first, containing CHR and HC subdirectories. Recommended, not enforced. 
# Allows for `list_transcripts()` class method to be used and more robust path-handling of transcripts.  
Transcript.set_directory_path("/data/transcripts")

# Iterate through all transcripts
for transcript in Transcript.list_transcripts():
    print(transcript.filename)
    print(len(transcript.participant_lines))

# Define transcript objects
t1 = Transcript("CHR/subject_001/interview.txt")
t2 = Transcript("HC/subject_042/interview.txt")

# Access properties
print(t1.patient_id)             # PatientID string when filenames in typical AMP SCZ format
print(t1.group_status)           # ClinicalGroup.CHR
print(len(t1.lines))             # Total line count
print(len(t1.participant_lines)) # Participant utterances only
print(len(t1.interviewer_lines)) # Interviewer utterances only

# Iterate over lines
for line in t1.participant_lines:
    print(f"{line.line_number}: {line.timestamp} - {line.text}")
```

### `combine_data.py`

Used to combine efficiently combine CSVs with `patient_id` columns, and optionally, `clinical_status`, appending all other columns afterwards. Uses Polars to support memory-efficient loading and speedy operations.

``` python
from utils.combine_data import combine_csvs

# Expects CSV with `patient_id` column, optionally `clinical_status`, and then metric columns
c1 = "some/path/data.csv" # `patient_id`, `metric_1`
c2 = "some/path/data.csv" # `patient_id`, `clinical_status`, `metric_2`

# Results in combined Polars DataFrame with `patient_id`, `clinincal_status`, `metric_1`, `metric_2`, matched on `patient_id`
combined_data = combine_csvs(c1, c2)
```

## Using `cli/`

### `assign_roles.py`

Classifies PARTICIPANT and INTERVIEWER speaker tags using vLLM batch inference.

``` bash
python cli/assign_roles.py --i "input/transcripts/" --o "output/" --thinking "medium" --gpu 0 --tp 1 --batch-size 32
```

### `assign_interview_type.py`

Classifies interview type as OPEN or PSYCHS using vLLM batch inference.

``` bash
python cli/assign_interview_type.py --i "input/transcripts/" --o "output/" --thinking "medium" --gpu 0 --tp 1 --batch-size 32
```

### `structure_transcripts.py`

Organizes transcripts by language and clinical status. Optionally uses CSV for clinical status mapping.

``` bash
python cli/structure_transcripts.py --i "input/transcripts/" --o "output/" --text-type "open" --gpu 0
python cli/structure_transcripts.py --i "input/" --o "output/" --text-type "psychs" --csv "status.csv" --gpu 0
```
