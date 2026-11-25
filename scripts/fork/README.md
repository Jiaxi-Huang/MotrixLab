# Fork integration

This fork keeps upstream code and fork-only automation on `main`. Feature combinations live on separately tested `main-*` integration baselines. `upstream/main` is the clean reference for independent topic branches; never base an upstream contribution on an integration baseline. Update main by merging upstream/main, preserving the automation commits. Each integration also snapshots upstream/main directly, so it does not depend on a manual main sync to pick up upstream changes.

`scripts/fork/integrations.json` maps each integration baseline to its ordered `topics` and declared `downstream` branches. `main-supermimic` combines the three active generic topics and supports downstream `supermimic`. Add another `main-xxxx` entry to define a separate combination; a topic may participate in multiple combinations. Each combination has independent candidates, validation, publication and concurrency. A failure in one does not block the others. Removing a topic from the list does not remove already merged code: explicitly reconcile or revert that feature on the integration branch.

Each active topic contains exactly one commit on top of upstream/main, with only its own functionality. Squash edits and upstream updates into that commit before publication; do not merge an integration baseline or downstream branch back into a topic. Verify `git rev-list --count upstream/main..TOPIC` is 1 and its parent is upstream/main. Published topic updates rewrite history and require an explicit lease against the previously fetched tip. Updates to shared files may require explicit merge resolution, even when features are independent.

The fork-only **Fork integration** workflow polls every ten minutes (GitHub may delay scheduled runs), and can be dispatched manually. It snapshots upstream/main, the selected integration baseline, fork/main and its configured topic tips, rejects rewritten topic history, then merges into a temporary candidate. It validates the exact candidate SHA with Python 3.10–3.13 fast/slow tests, fork orchestration tests and pre-commit hooks. It publishes with an ordinary fast-forward push only after validation and a second source-ref check. Unchanged inputs skip validation. It never creates an upstream PR or pushes upstream.

The workflow owns its checks because pushes made using `GITHUB_TOKEN` do not trigger the usual push workflows. Only this workflow should advance each integration baseline while its validation runs are active. Different baselines run independently; updates to the same baseline are serialized. Concurrent source updates fail publication and are picked up by the next run. A very small race between the final read and push cannot be made atomic across several Git refs; the integrated ledger always identifies the exact tested sources.

Merge conflicts stop publication. Download the `candidate` artifact for source revisions, merge diagnostics and a Git bundle. Resolve on a fork branch based on the current integration baseline (or fork/main for a new baseline), merge every source revision, and dispatch with `baseline` set to the integration branch and `resolved_candidate` set to the resolved branch name. The tool verifies ancestry and runs all checks again. Temporary candidate refs are deleted after the run; artifacts remain for 14 days. Initial setup uses this same path after installing the workflow on main. The scheduled dispatcher lives on main, as GitHub schedules run from the default branch. Select a `baseline` for manual runs, or leave it empty to process all configured baselines. No automatic conflict strategy is used.

For a manually validated update of supermimic:

```bash
python scripts/fork/supermimic.py \
  --baseline main-supermimic \
  --baseline-sha <full-current-origin-main-supermimic-sha> \
  --python /absolute/path/to/torch211-environment/bin/python \
  --motion-file /absolute/path/to/cached/motion.npz \
  --publish
```

Omit `--publish` to validate without publishing. The command retains its isolated worktree and logs under `/tmp/supermimic-validation-*`. It merges the selected baseline, runs non-integration tests and APG integration tests, exercises a real CPU simulation/learning update and checkpoint restore, then runs hooks. The APG environment currently needs Torch 2.11; the generic baseline retains its locked dependencies. Ensure `prek` is on PATH. Review skipped tests; a missing hardware runtime is not hardware validation. Successful publication only advances origin/supermimic and does not move a developer's active checkout.

The initial migration is exceptional: retain backup refs, rebuild the APG documentation and code commits on the integrated baseline, validate, then replace supermimic with an explicit `--force-with-lease=<ref>:<old-sha>`. Subsequent updates use the merge command above. Before eventually opening upstream PRs, verify each topic independently against current upstream/main. When upstream absorbs a topic, retire it from the relevant integrations.json topic lists after verifying equivalent behavior and reconciling each affected baseline; never silently drop a source merely because its branch disappeared.

Run orchestration contract tests explicitly:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest scripts/fork/tests -m integration
```

The `fix/walk-reset-randomization` source was retired after upstream PR #177 addressed the same compile-time randomization gates. Its merged history remains in the integration baseline; reconcile overlapping reset changes when incorporating upstream #177.

## Maintaining one commit per topic

The scheduled integration workflow deliberately rejects rewritten topic ancestry. Do not bypass that check or publish rewritten topics alone. For a history-only squash, verify each new topic tree exactly matches its previously tested tip, retain backup refs, and build a reconciliation candidate from the current integration baseline. Record the new topic commits as merged ancestors and update its integrated-source ledger. Confirm that reconciliation changes no code, run hooks, then atomically publish the rewritten topics (with explicit leases) and the fast-forward integration baseline. Merge that baseline into its downstream branches afterward.

If squashing also changes code or incorporates a newer upstream tree, the previous test results do not validate the new tree: resolve and validate the integration candidate before publishing the coordinated update. Apply this reconciliation to every configured baseline consuming the rewritten topic. A normal developer commit may be used while working locally, but the published topic must retain the one-commit invariant.

The motion-curricula and worker-cpu-budget sources are provided by upstream main through PRs #179 and #180. The read-freshness source is retired; integration and downstream branches use upstream read behavior while an engine-side replacement is pending.
