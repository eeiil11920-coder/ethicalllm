# Anonymisation and blinding protocol — expert ratings, wave 2

Purpose: prevent brand/model expectations from biasing expert ratings, and
allow the ratings to be archived and shared without exposing rater identity.

## Blinding of model identity (before rating)

1. `make_sheets.py` strips every provider/model identifier from the
   transcripts (provider names, model ids, dry-run tags, API metadata).
2. Each provider is replaced by a neutral code **M1–M5**. The code
   assignment is **randomised per rater** (seeded), so raters cannot align
   codes with each other.
3. Dialogues are identified only by a `DialogueCode` (e.g. `D047`); the
   mapping DialogueCode -> (provider, case, condition) is stored in
   `output/provider_key_<rater>.csv`, kept by the corresponding author,
   **never** distributed to raters until all sheets are returned.
4. Transcripts are presented in a per-rater randomised order to dilute
   order and fatigue effects.
5. A regex pass removes residual self-identification in the responses
   ("As ChatGPT/Claude/Gemini/Grok/DeepSeek…", "I'm an AI developed by
   OpenAI/Anthropic/Google/xAI/DeepSeek…") replacing it with "As an AI
   assistant…". Residual stylistic cues cannot be fully removed; this is
   acknowledged as a limitation in the manuscript.

## Anonymisation of raters (after rating)

1. Raters are referred to as R1, R2, R3 in all archived files and in the
   manuscript; the name–code correspondence is kept offline by the
   corresponding author.
2. Free-text comments are screened before archiving and any
   self-identifying remark is redacted.
3. The public repository contains: the blinded rating sheets, the blinding
   key (released only after consensus is finalised, as it is needed to
   reproduce the analysis), and the consensus scores. No personal data of
   the raters is archived, in line with journal data-availability policy.

## Integrity rules

- Raters must not query any LLM while rating, and must not attempt to
  re-identify the model behind a transcript.
- Inter-rater reliability (Krippendorff's alpha per dimension) is computed
  on the blinded, pre-consensus sheets; disagreements >= 3 points trigger
  discussion in the consensus meeting; the consensus sheet is a separate
  file, never an overwrite of the originals.
