# Resume Category Classification — Traditional ML

## Add the dataset
Place your CSV in this folder and name it `Resume.csv`. The training script looks for common text columns (`Resume`, `Resume_str`, `Resume_Details`, `resume`, `text`) and target columns (`Category`, `category`, `Job_Category`, `label`). If your names differ, update the candidate lists in `train_model.py`.

## Setup (VS Code PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, switch the VS Code terminal to Command Prompt and use `.venv\Scripts\activate.bat`.

## Train and evaluate
```powershell
python train_model.py
```
This performs basic cleaning, removes exact duplicate text-label rows and ambiguous resume texts with conflicting labels, compares six traditional models using 5-fold cross-validation macro F1 on the training split, evaluates the selected model once on a held-out test set, then saves the complete TF-IDF + classifier pipeline.

## Run Streamlit
```powershell
streamlit run app.py
```

## Deployment
Push the code to GitHub and deploy `app.py` using Streamlit Community Cloud. The model artifact is generated locally by training. For deployment, include `models/resume_classifier.joblib` in the repository if its size and data policy allow, or configure a model artifact download/setup step. Never commit private resumes or sensitive personal data.

This project uses traditional ML only. No accuracy is guaranteed; performance depends on dataset quality and the train/test split.
