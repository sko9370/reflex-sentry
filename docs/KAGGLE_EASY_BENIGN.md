# Score the 300 easy-benign prompts

Local input is ready at `data/processed/easy_benign_for_teachers.parquet`.
It contains 300 selected ToxicChat prompts; neither existing teacher score
file contains these ids. The paired gold slice is
`data/processed/test_easy_benign.parquet`. It has not yet been appended to
`test.parquet`, so the current comparison still uses the same test population
for all models.

1. Add `easy_benign_for_teachers.parquet` to the private Kaggle dataset used
   by the existing teacher notebook. Enable its GPU and use the existing
   repository/dependency setup and Hugging Face access configuration.
2. From the repository directory in that notebook, run this cell. Change
   `INPUT` if Kaggle mounts your new dataset version at a different path.
   Each teacher runs in a separate process so its GPU allocation is released
   before loading the next one. Existing output files support resumption.

```python
import os
import subprocess
import sys
from pathlib import Path
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
INPUT = Path("/kaggle/input/datasets/sko9370/parquets/easy_benign_for_teachers.parquet")
assert INPUT.is_file(), INPUT

for model in ("llama_guard_3_8b", "qwen3guard_gen_8b"):
    subprocess.run([
        sys.executable, "-m", "reflex_sentry.teacher.score",
        "--model", model, "--inputs", str(INPUT),
        "--out", f"/kaggle/working/teacher_scores_{model}_easy_benign.parquet",
        "--batch-size", "8", "--max-length", "512", "--load-in-4bit",
    ], check=True)
```

3. Download both output parquets into local `data/interim/`:
   - `teacher_scores_llama_guard_3_8b_easy_benign.parquet`
   - `teacher_scores_qwen3guard_gen_8b_easy_benign.parquet`

After they are returned, validate that each covers all 300 expected ids,
merge them with the corresponding existing teacher scores, append the saved
easy-benign slice once, and regenerate all models' test predictions/reports.
This is inference only; Stage B does not need retraining for the added slice.
