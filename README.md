# Science Fair 2026
<h3> Authors: Grey Sparks & Daniel Chen</h3>

# Stack
* **Core**
* * Python

* **Networking**
* * [Junction](https://github.com/greysparks00/junction) 


* **Dataset(s)**
* * [UT-MoDaPark v6](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GEKIVC)

# Running
This project is built using **[uv](https://docs.astral.sh/uv/getting-started/installation/)** and **[StreamLit](https://streamlit.io/)**.

**1.** Run the command `uv sync` in the terminal. This will download all the required packages

**2.** Run the command `uv run python train.py` in the terminal. This will train the local model on the dataset.

**3.** Run the command `uv run streamlit run dashboard.py` in the terminal. This will start the webserver which displays the results each model's run metrics. This is not live updated, so you must run this command everytime after you train and run a new model.

# Citations

Moradi, Hadi. 2025. “UT-MoDaPark Dataset.” Harvard Dataverse. https://doi.org/10.7910/DVN/GEKIVC.