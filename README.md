# Two-step heart disease screening (v3)

Class assignment, Advanced Machine Learning, Strathmore University. Dataset: Heart Failure Prediction (Kaggle, 918 patients, 11 inputs).
All numbers in the paper come from `results/*.csv`, produced by the scripts here.

Live app: https://two-step-heart-screening.streamlit.app/ (research prototype, not for clinical use)

## Folders
- `scripts/`  `common.py` (settings, models, calibration, thresholds), `02_train_models.py`, `03_ladder.py`, `04_evaluate.py`, `05_site_validation.py`, `06_export_models.py`, `run_all.py`
- `app/`      Streamlit application (`app.py`, `logic.py`); trained models are written to `app/models/`
- `tests/`    26 tests (pipeline and application)
- `results/`  tables (CSV), `summary.md`, `run_log.txt`
- `figures/`  figures used in the paper
- `paper/`    paper (.docx and .pdf) and its generator (`paper/source/`)
- `docs/`     `DECISIONS.md` (design decisions and the rules fixed before the results)
- `external/` UCI source files used only by the site validation
- `dataset/heart.csv`

## Run (macOS or Linux, Python 3.10 to 3.13)
`run_local.sh` copies the project to `~/hf_run`, creates its own virtual environment there (never inside a synced folder), installs the pinned `requirements.txt`, runs, and copies results and figures back.

```
V3="<path to this v3 folder>"
bash "$V3/run_local.sh"            # full run: about 11 minutes on a 12-core Mac
bash "$V3/run_local.sh" --test     # 26 tests
bash "$V3/run_local.sh" --export   # re-export the models used by the app
bash "$V3/run_local.sh" --app      # start the web app at http://localhost:8501
bash "$V3/run_local.sh" --site     # leave-one-site-out validation only
```
The app runs in the foreground; press Ctrl+C or close the terminal to stop it. To keep it running after closing the window: `nohup bash "$V3/run_local.sh" --app > ~/hf_run/app.log 2>&1 &`.

## Protocol in one paragraph
3 x 10 stratified outer folds, 5-fold inner grid search on accuracy, seed 42. Stacking, calibration and cut-offs use inner out-of-fold predictions of the training fold only. One 80/20 hold-out is also reported. Prevalence scenarios (2, 5, 10, 20 %, and the dataset's 55.3 %) use the prior-shift correction of Saerens et al. (2002).

## Rebuild the paper
```
cd paper/source
python3 make_numbers.py && python3 make_equations.py && python3 make_figures.py
node build_paper.js ../HeartDisease_TwoStep_Screening_Paper.docx     # needs the Node package "docx"
```

## Limits
Research prototype, not for clinical use. Single historical dataset; no clinical validation.

## Publish on GitHub and deploy the app
```
bash publish_github.sh                 # builds ~/hf_github/two-step-heart-screening and pushes a PRIVATE repo (needs gh, logged in)
bash publish_github.sh my-repo-name --public
```
To host the app online, use Streamlit Community Cloud (share.streamlit.io): sign in with GitHub, choose the repository, branch `main`, main file `app/app.py`, Python 3.12 (Advanced settings). It installs the pinned `requirements.txt`; the trained models in `app/models/` are part of the repository. GitHub Pages cannot run this app, because it needs a Python server.
