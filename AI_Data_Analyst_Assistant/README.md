# AI Data Analyst Assistant

A local Streamlit dashboard for CSV and Excel data profiling, interactive charts,
approved cleaning, and natural-language analysis. Pandas computes every numerical
answer. Ollama runs locally and is used only to select a supported operation and
explain its verified result; raw uploaded rows are never sent to the model.

## Requirements

- Python 3.9 or newer
- Ollama installed locally for natural-language analysis (profiling, cleaning,
  charts, and downloads also work without Ollama)

## Install

From this directory, create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, call the environment executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Configure Ollama

Install Ollama, start its local service, and pull a model, for example:

```powershell
ollama pull llama3.2
```

By default, the app makes real non-streaming requests to
`http://localhost:11434/api/chat` and uses the `llama3.2` model. Requests time out
after 120 seconds with an actionable message; connection failures and missing
models are reported in the analyst panel.
Optional environment variables configure the local endpoint and model:

```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
$env:OLLAMA_MODEL = "llama3.2"
```

For privacy, the application rejects non-loopback Ollama hosts. It sends only
column names/types and the user's question for operation selection, followed by
Pandas-computed result rows for explanation. It never sends uploaded row data or
calls a paid/external AI API.

## Run

```powershell
streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit, upload a `.csv` or `.xlsx` file in the
sidebar, and choose a section from the sidebar navigation.

## Features

- Dataset preview, row/column counts, missing/duplicate counts, column types and
  cardinality, and numeric descriptive statistics.
- Interactive bar, line, and scatter charts, category filters, and date ranges.
- Ollama-assisted question interpretation with a strict allowlist of Pandas
  aggregate, group, top-category, and monthly-trend operations. Totals,
  averages, highest-sales products, and regional comparisons are calculated
  directly from the data with Pandas.
- Data-quality checks and cleaning previews. Changes to the working dataset
  happen only after the user presses **Approve and apply changes**.
- CSV downloads for the current (possibly cleaned) dataset and analysis tables,
  plus a downloadable business summary report.
- `sample_sales.csv` for trying the dashboard without preparing a file.
