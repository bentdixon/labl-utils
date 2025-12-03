## Instructions

`cd your-project`

`git clone https://github.com/bentdixon/labl-utils/tree/main`

### Using **TranscriptUtils**

`from utils.transcript import Transcript, ClinicalGroup`

`Transcript.set_directory_path("/data//transcripts") # Must set the transcript directory first, containing CHR and HC subdirectories`

`t1 = Transcript("CHR/subject_001/interview.txt") # Define transcript objects`

`t2 = Transcript("HC/subject_042/interview.txt")  # Define transcript objects`

`print(t1.group_status) # ClinicalGroup.CHR` `print(len(t1.lines)) # Total line count`

`print(len(t1.participant_lines)) # Participant utterances only`

`print(len(t1.interviewer_lines)) # Interviewer utterances only`

`for line in t1.participant_lines: print(f"{line.line_number}: {line.timestamp} - {line.text}")`