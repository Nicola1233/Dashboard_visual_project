# Dashboard 

This project is a **interactive dashboard built with Streamlit and Plotly**.
It visualizes daily life of a BDMA student data such as steps, sleep, and other data from a CSV file.

The app is written in Python and can be run **with Anaconda (conda)** or **without Anaconda (pip)**.

---

## Project structure
```text
pyinterface-dashboard/
├─ app.py # Streamlit main app
├─ charts.py # Plotly charts
├─ styles.css # Custom CSS
├─ environment.yml # Conda environment
├─ requirements.txt # Pip dependencies
└─ data/
   └─ Dataset_User1234.csv
```
## Run code

### With Anaconda
```bash
conda env create -f environment.yml
conda activate pyinterface-dashboard
streamlit run app.py
```
### Without Anaconda
```bash
pip install -r requirements.txt
streamlit run app.py
```
