# SPEC: <experiment name>

Fill this in and commit it *before* writing the implementation for that experiment.

## Question
What exactly is being tested? One falsifiable sentence.

## Method
What runs, on what data, compared against what control/baseline.

## Metric
What is measured, and what threshold/comparison decides the outcome. State this before
seeing results.

## Instrument characterization
Before spending compute: what properties of the measuring instrument (filter, model,
comparison test) are already known or computable in advance that could void the result?
(e.g. does the metric have enough headroom on this data to detect anything at all?)

## Interpretation
What result would count as CONTRIBUTES / NULL / INCONCLUSIVE, decided before the data is
seen.

## Amendments
Dated notes only — do not rewrite the sections above after the experiment starts.
