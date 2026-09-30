# BF16 query configuration study

These assets preserve the observed complete stock BF16 dispatch table and the
measured query-only configuration overlay. They live outside the installed
configuration directories and **do not enable any production route**. The
configuration-change commit can be compared with its baseline-snapshot parent.

The study uses gfx950 with 256 CUs, BF16 canonical operands, BF16 output, and
N=K=2048. Only six exact dispatch keys and four existing JSON M buckets change.
All other CSV lines and JSON configuration values remain unchanged. JSON
whitespace differs because the experimental writer uses two-space indentation.
The full-table snapshot preserves the real merged stock choices rather than
inferring a baseline by deleting candidate rows.

CSV `us` values in the measured overlay are source-bound clean **warm graph**
measurements from this research protocol, not native-tuner or profiler timings.
They do not assert a production serving gain. The separate family-kernel study
is not part of this configuration change. Numerical qualification, whole-block
transfer, inactive cells and timing quarantine remain separate evidence.

`provenance.json` binds the original stock assets, exact measured asset bytes,
selected configurations and their qualification summary hashes.
