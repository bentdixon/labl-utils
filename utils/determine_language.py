from typing import cast
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
from utils.transcripts import Transcript
from data.langs import Language

nlp = Pipeline(lang="multilingual", processors="langid")

def determine_language(transcript: Transcript) -> Language:
    text = "\n".join(line.text for line in transcript.lines)
    doc = cast(Document, nlp(text))
    try:
        if doc.lang:
            return Language[doc.lang] 
        else:
            print(f"Language for {transcript.patient_id} could not be identified")
            return Language.UNKNOWN
    except KeyError as e:
        print(f"Error {e}")
        return Language.UNKNOWN

