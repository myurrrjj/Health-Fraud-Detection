# Medicare Provider Risk Explorer

A reproducible Streamlit demonstration that estimates whether provider-level summaries resemble providers labelled PotentialFraud in the supplied Medicare claims dataset.

**Maintainer:** Mayur Jindal

## What it does

- Combines provider labels, inpatient claims, outpatient claims, and beneficiary attributes.
- Creates one training row per provider, matching the provider-level target label.
- Trains a LightGBM classifier and evaluates it on a held-out provider split.
- Opens a Streamlit form for provider-level inputs and displays the model score and local tree contribution values.

This is an educational prototype. A model score is not a finding of fraud and must not be used for real claims decisions.

## Data files

Place these four source files in data/raw/:

- Train-1542865627584.csv
- Train_Beneficiarydata-1542865627584.csv
- Train_Inpatientdata-1542865627584.csv
- Train_Outpatientdata-1542865627584.csv

The matching Test CSVs are optional; the current training pipeline does not use them. Dataset files are excluded from Git. Obtain the data from its original source and follow its terms before sharing or redistributing it.

## Requirements

Use 64-bit CPython 3.11 through 3.14. Dependency versions are pinned in the project metadata.

### Linux or macOS

From the repository root:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

### Windows PowerShell

From the repository root:

~~~powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

If your installed Python is not 3.14, replace -3.14 with an installed version from 3.11 through 3.14.

## Run the app

From the repository root, with the virtual environment active:

~~~bash
streamlit run app/app.py
~~~

On first launch, the app trains the model from the four files in data/raw/ and saves it to models/provider_fraud_model.joblib. Later launches reuse that model. To train it manually:

~~~bash
python -m fraud_detection.train
~~~

The application gives a clear message if source files are missing.

## Run the notebooks

Install the optional notebook environment:

~~~bash
python -m pip install -r requirements-notebooks.txt
~~~

Then open the notebooks from the repository with Jupyter. The notebooks call the same functions used by the app and training command.

## Project layout

~~~text
app/                         Streamlit interface
src/fraud_detection/          Data preparation and model training
notebooks/                    Exploration, feature review, and training walkthrough
data/raw/                     Local source CSVs; ignored by Git
data/processed/               Reserved for local generated data; ignored by Git
models/                       Locally generated model; ignored by Git
~~~

No generated tables or model files are required in the repository. The model is recreated from the source CSVs.

## Model and data limitations

The target is the dataset's PotentialFraud provider label. It is a provider-level label, not adjudicated claim-level ground truth. Evaluation splits providers so the same provider cannot appear in both the training and held-out sets. The resulting metrics describe this dataset and should not be treated as evidence about real-world providers.
