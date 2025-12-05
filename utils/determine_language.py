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
        return Language[doc.lang] if doc.lang else Language.UNKNOWN
    except KeyError:
        return Language.UNKNOWN