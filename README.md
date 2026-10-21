# Battery dispatch and settlement calculations

Python code for battery dispatch, capacity selection, demand charges, coincident peak settlement, and operating-emissions calculations. The repository contains calculation code and a data downloader. Input datasets and computed results are generated locally and are not included.

## Setup

Use Python 3.12 and run commands from the repository root. Create a virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

The linear programs use SciPy's HiGHS solver. NumPy, pandas, matplotlib, and tzdata support numerical processing, data preparation, plots, and time-zone conversion. The CityLearn simulator is not required.

## Download the main inputs

```bash
python code/download_data.py
```

This downloads 22 files from a fixed CityLearn source revision, checks their SHA-256 hashes, and creates the local `data/citylearn2022/` manifest and output directories. Internet access is required. The source license is downloaded with the data; dataset terms are separate from the repository's code license.

## Run the main calculation chain

Run these commands in order:

```bash
python code/run_experiment.py scale --workers 1
python code/tea.py
python code/extensions.py
python code/demand_charge.py
python code/coincident_peak.py
python code/build_d10_dispatch.py
python code/revision_v6_checks.py
python code/v6_attach.py
```

| Entry point | Inputs and purpose |
| --- | --- |
| `run_experiment.py scale` | Reads the downloaded building profiles; computes rule-based and LP dispatch results, the capacity grid, and reference baselines. Creates `raw_metrics.csv` and `results.json`. |
| `tea.py` | Reads `raw_metrics.csv`; computes lifetime costs and NPV. |
| `extensions.py` | Calculates peak limits, terminal state-of-charge cases, and export-credit sensitivity. Creates `extensions.json`. |
| `demand_charge.py` | Requires `raw_metrics.csv` and `extensions.json`; recalculates peak-constrained cases and computes individual demand-charge scenarios. It also updates `extensions.json`. |
| `coincident_peak.py` | Recomputes demand-aware dispatch from the building profiles and evaluates coincident peaks across groupings. |
| `build_d10_dispatch.py` | Solves 34 annual LPs and creates a derived dispatch archive with a hash manifest. |
| `revision_v6_checks.py` | Reads the derived dispatch archive, demand-charge results, and cost-model outputs; calculates settlement and lifetime-cost checks. |
| `v6_attach.py` | Reads the main results and settlement summaries, replays the PRICE rule, and adds the resulting metrics to `results.json`. |

Keep `extensions.py` before `demand_charge.py`: running it afterward overwrites the corrected peak-limit summary. The final two commands require outputs from the preceding stages; they do not work on a clone containing source code alone. Once those inputs exist, they can be rerun without solving the LP or MPC experiment grids.

An optional smaller calculation is available before the full chain:

```bash
python code/run_experiment.py pilot --workers 1
```

The pilot still solves optimization problems and writes a separate pilot result. It does not generate the files required by the full calculation chain.

Full source computation is substantially more expensive than reading saved results. `--workers` controls only `run_experiment.py`; other calculation scripts have their own multiprocessing defaults. Runtime depends on CPU resources and solver behavior. Generated files are placed under `data/`, `results/`, and `outputs/`.

## Optional calculations

### Threshold sensitivity and receding-horizon control

After the main scale calculation:

```bash
python code/threshold_sensitivity.py
python code/run_mpc.py
```

Threshold sensitivity reads the main dispatch metrics. The MPC run repeatedly solves hourly receding-horizon problems and can take considerably longer than the other stages. Its worker count is set within the script; it does not accept the `run_experiment.py --workers` option.

### Marginal operating emissions

After downloading the main inputs:

```bash
python code/build_marginal_data.py
python code/marginal_carbon.py
```

The builder retrieves SGIP operating-emissions data and source metadata over the network, processes the five-minute series into hourly inputs, and records provenance and checksums. Download availability and provider terms apply. The calculation uses a bounded overlap with the building profiles and includes alternative clock alignments and model versions. These source files and computed outputs are not bundled.

The first run uses the provider's live download. Later runs check against the locally saved summary. The repository does not contain a frozen SGIP signal archive, so upstream changes can affect byte-level reproducibility.

### External constructed case

Obtain the provider input files separately and place them at these paths:

```text
data/external/raw/household_power_consumption.txt
data/external/raw/4466719_25.92_117.73_2020.csv
```

The first is the UCI Individual Household Electric Power Consumption input; the second is the NSRDB weather/solar input expected by the preparation script. The filenames and schema must match the script's inputs. The main scale calculation must already have generated `results/results.json`.

```bash
python code/build_external_data.py
python code/external_case.py
```

This branch combines independently sourced load and solar/weather inputs into a constructed case. It is not a paired field observation of one site.

## Source layout

`core.py` implements shared profiles, rule-based dispatch, metrics, and linear programs. `baselines.py` provides reference comparisons, and `mpc.py` supplies receding-horizon optimization. The remaining files are the preparation and calculation entry points described above. Reproducing a calculation requires obtaining its inputs and running its dependencies.
