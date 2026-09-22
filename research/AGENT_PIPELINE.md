# The role-rotating research pipeline

This describes how `research/GIT_WORKFLOW.md`'s process runs in practice when **one** Claude
Code session carries it out, picking up whichever of three roles — researcher, implementer,
reviewer — GitHub state says is next, and coordinating only through that state (issues,
branches, PRs, labels), never through in-session memory of "what I decided as researcher an
hour ago."

Adapted from the same pattern used in this account's `Replica-Recall-Divergence` repo — the
role definitions and role-selection logic are domain-agnostic, so they transfer directly;
only the domain-specific examples below (branch types, what counts as an "experiment") are
this project's own.

## Why GitHub state instead of live coordination

Every handoff between roles has to be a durable, auditable artifact — the same reason
`DECISION_LOG.md` exists. This matters even more because one session plays every role:
without a hard rule to re-derive each role's state from GitHub rather than from what the
session remembers writing as a different role minutes earlier, the roles quietly merge into
one voice and the separation of concerns (proposer / builder / independent checker) stops
meaning anything. Issues and PR comments are the coordination layer, full stop — not context
carried over in the conversation.

## Roles

### Researcher

Finds the next real gap, writes it up as a pre-registered spec, files it as an issue. Does
not implement anything.

**Where to look for the next question:**
1. The top-level `README.md`'s findings tiers (HYPOTHESIS and OPEN entries are exactly the
   standing list of unanswered questions).
2. `research/README.md`'s experiment index — what's already open, in progress, or closed.
3. `research/DECISION_LOG.md` — so a question already investigated and ruled out doesn't get
   re-proposed as new.
4. `research/RELATED_WORK.md` — in case the question is already answered by existing
   literature rather than needing a new experiment.

**What makes an issue correctly scoped:** one answerable question. "Does Arm C beat the
classical arms on degraded ridges?" is scoped; "does deep learning work better?" (unboundedly)
is not. A question with an obvious, already-known answer isn't worth a branch — that's what
makes steps 1-3 above load-bearing, not a formality.

**Filing:** use the `research-question.yml` issue form (New Issue → Research question).
Every field maps directly onto `research/SPEC_TEMPLATE.md` — the issue and the spec should
never diverge into two documents saying different things. Label with the correct `type:*`
and leave `stage:proposed`.

**Do not:** implement anything, open a branch, or write code. If tempted to prototype "just
to check feasibility" first, that's a signal the issue needs a feasibility/confound note in
its own fields, not code.

### Implementer

Claims one open issue, builds it on a branch, opens a PR.

**Claiming:** find an issue labeled `stage:proposed`, self-assign it, swap the label to
`stage:claimed`.

```bash
gh issue list --repo shlokkvaishnav/ISRO-Ridge-Detection-System --label stage:proposed
gh issue edit <N> --add-assignee @me --add-label stage:claimed --remove-label stage:proposed
```

**Branch naming:** follow `research/GIT_WORKFLOW.md`'s prefixes exactly (`research/<topic>`,
`experiment/<name>`, `analysis/<name>`, `method/<name>`, `reproduction/<target>`) — do not
append the issue number to the branch name. The link to the issue lives in two places
instead: the first commit's `SPEC.md` (copy the issue body in verbatim, don't paraphrase),
and the PR's `Closes #<N>`.

**Implementing:** per `GIT_WORKFLOW.md` — spec first (already have it, from the issue), then
implementation, then validation, then the actual experiment. Update `SPEC.md`'s
Results/Interpretation/Decision sections once there's something to put there.

**If the branch adds a `research/<name>/` directory,** add its row to `research/README.md`'s
experiment index in the same PR — investigation, location, type, status.

**Opening the PR:** `.github/PULL_REQUEST_TEMPLATE.md` auto-populates — fill in every section
honestly, including a self-assessed MERGE/ARCHIVE/REVISE/ABANDON/REPRODUCE decision. Label
the PR `stage:in-review`, and swap the linked issue to `stage:in-review` too.

**Do not:** merge, invent scope beyond what the issue specified (a genuinely different,
better question found along the way is a new issue, not silent scope creep), or skip a
template section because the answer is inconvenient — an honest "confounds remain: X" is the
point, not a failure.

### Reviewer

Reviews the PR against `research/GIT_WORKFLOW.md`'s actual merge criteria — scientific
relevance, correctness, experimental validity, reproducibility, documentation,
interpretation, research integrity, integration, evidence. Comments, doesn't push code,
doesn't merge.

**Process:**
0. Count the review rounds already on this PR (its comment history is the record). At
   **three**, do not open a fourth: post the findings, leave the PR
   `stage:changes-requested`, and say explicitly a human should look. Do **not** approve at
   the cap.
1. **Audit the diff mechanically before reading any prose.**
   ```bash
   H=$(gh pr view <N> --json headRefOid -q .headRefOid)
   git fetch origin && git diff --name-status origin/master...$H
   git diff --name-status origin/master...$H | grep -E '^(D|R)'
   git diff --name-only  origin/master...$H | grep -E '(^|/)results?(_|/)'
   ```
   Any file the PR body doesn't account for, any deletion, and any change under a `results*/`
   directory is a finding before a word of the writeup has been read.
2. **Recompute every headline number from the raw committed data**, not the branch's own
   analyser — e.g. re-derive a recall percentage from the tile manifest and ground-truth
   catalog directly, not by trusting the PR's own summary script.
3. **Look for a derived quantity under which the effect disappears.** Different denominator
   (per-tile vs aggregate), different baseline (synthetic vs real), a rate instead of a raw
   pixel count.
4. Read the issue, the PR template's filled-in answers, and the diff, as if seeing them for
   the first time — see "Guarding role separation," below.
5. Check the mini-peer-review questions were actually answered, not just present.
6. Check `GIT_WORKFLOW.md`'s "when not to merge" list explicitly.
7. Post a PR comment with your own MERGE/ARCHIVE/REVISE/ABANDON/REPRODUCE recommendation and
   why. If REVISE or CHANGES REQUESTED, be specific enough the implementer role can act on it
   from the comment alone. **A MERGE decision must name the head SHA it reviewed.**
8. Label accordingly: `stage:changes-requested` if more work is needed,
   `stage:approved-pending-merge` if you'd merge it — label **both** the PR and its linked
   issue.
9. If ARCHIVE/ABANDON/REPRODUCE — or if the decision says to stop a line of work — label
   accordingly (PR and issue both), and add an entry to `research/DECISION_LOG.md` at this
   point, not left for later.

**Do not:** merge (that's the user's call), rewrite the implementer's code, or approve
because the numbers look good without checking whether the numbers answer the actual
question.

## Role selection (chain until idle, not one-role-per-tick)

Run these `gh` queries in order and play the **first** role whose condition is true. A single
loop iteration **chains**: the moment a role's one unit of work finishes, immediately re-run
this same query list from the top and play whatever fires next. Only stop once condition 5
(idle) is reached.

0. **`gh pr list --state merged --label stage:approved-pending-merge`** returns anything →
   post-merge verification, before any role plays. For each, take the head SHA named in the
   reviewer's MERGE comment and check `git merge-base --is-ancestor <sha> origin/master`. If
   it holds: relabel PR and linked issue `stage:merged`, update `research/README.md`'s
   experiment index. If not: the human merged a different head than was reviewed — comment
   naming both SHAs, label `stage:changes-requested`.
1. **`gh pr list --label stage:in-review`** returns anything → play **Reviewer** on the oldest
   PR.
2. Else, **`gh pr list --label stage:changes-requested`** returns anything → play
   **Implementer** on the oldest such PR: act on the review comment, push fixes, relabel back
   to `stage:in-review`.
2b. Else, **`gh issue list --label stage:claimed --assignee @me`** returns anything → play
   **Implementer** on it — work already claimed and not yet a PR. If stale (no branch, no
   commits), un-assign and relabel `stage:proposed`.
2c. Else, if **`gh pr list --label stage:approved-pending-merge`** returns **more than one**
   PR → idle for this tick (log it as condition 5). This does not stop queries 0, 1, 2, or 2b.
3. Else, **`gh issue list --label stage:proposed --json number,assignees -q '.[] | select(.assignees | length == 0) | .number'`**
   returns anything → play **Implementer** on the oldest unclaimed issue.
3b. Else, **ask what can be answered from data already committed**, before proposing anything
   that needs new tile extraction or new experiments. Concretely: is there a column or
   artifact in `data/tiles/manifest.json` or a prior run's saved masks that no study has
   analysed yet? A cheap analysis of existing data outranks a new sweep.
4. Else, **`gh issue list --label stage:proposed`**'s count is below threshold N (default
   N = 3) → play **Researcher** and file exactly one issue.
5. Else, idle: log which condition was checked and why nothing fired, then stop this
   iteration.

**Bounding the review/revision cycle.** The cap that bounds a PR cycling between the
Reviewer and Implementer roles is **step 0 of the Reviewer's instructions**, not a rule of
role selection.

**Chaining stops being safe to auto-continue past a MERGE-decided PR.** A MERGE-approved PR
is labeled `stage:approved-pending-merge`, removing it from query 1. Nothing downstream of
that label fires again until a human actually merges (see below) — that's intentional.

## Why merge stays manual

Every other step in this pipeline collapsed into one auto-chaining loop because each one is
independently checkable. Merge is different: its entire purpose is a check *external* to the
process that produced the thing being checked. GitHub's own `gh pr review --approve` refuses
a PR opened by the same account, a mechanical reminder that reviewer and implementer share
one identity here. A human clicking merge is the last checkpoint that isn't also this same
account grading its own homework.

## Guarding role separation within one session

- **Re-derive state from GitHub, not from this conversation's memory.** When playing
  Reviewer, read the PR, the linked issue, and `SPEC.md` on the branch as if seeing them for
  the first time — do not reuse implementer-role reasoning from earlier in the same session
  as a shortcut.
- **Never review or approve your own PR**, regardless of which session wrote it.
- **Never let the researcher role scope an issue around a solution the session already has
  in mind.** The researcher role's job is to find a real gap, not pre-stage easy work for the
  implementer role it's about to become.
- If a tick's role selection would have this session review its own immediately-prior work,
  that's expected and fine — the rules above are what keep it honest.

## Running this as a loop

Role selection above is exactly what a scheduled loop polls each tick — the harness's
`/loop` skill (dynamic pacing, no fixed interval needed) or a cron.

**The loop is on by default.** `.claude/settings.json` registers a `SessionStart` hook
(`.claude/hooks/pipeline-autostart.sh`) that tells every Claude Code session opened in this
repository to start `/loop` on this document's role selection on its first turn, unless the
user's message says not to. Disable it by removing the `SessionStart` entry.

**Changes to this pipeline itself go straight to `master`.** Edits to the roles, role
selection, templates, and hooks — the process, not the research — are committed to `master`
directly, the way `GIT_WORKFLOW.md` already treats editorial changes.
