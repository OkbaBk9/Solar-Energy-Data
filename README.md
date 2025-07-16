Solar Power Potential in the Algerian Sahara

This project analyzes solar irradiance data in Algeria's Saharan regions to identify optimal zones for solar farm investment.

Problem Statement :
Algeria's Sahara has immense untapped solar potential. This project aims to explore and quantify this potential using NASA POWER daily solar radiation data.

Dataset
- Source: NASA POWER API
- Locations: Tamanrasset, Adrar, etc.
- Features: Solar irradiance (DNI, DHI), temperature, clearness index, seasonal averages.

Methodology
1. Data Collection** via NASA POWER API.
2. Preprocessing**: Feature engineering (clearness index, temp category).
3. Exploratory Analysis**: Trends across time, regions.
4. Ranking**: Cities scored by solar viability.
5. Regression models to predict solar output.

Key Insights
- Adrar and Tamanrasset have the most consistent high irradiance.
- Summer months yield peak solar potential.
- Clearness index confirms stable conditions in southern Algeria.

Tech Stack
- Python (pandas, numpy, matplotlib, seaborn)
- Jupyter Notebooks
- NASA POWER API
- *(Optional)* Folium, Scikit-learn

Project Structure :

solar-sahara-project/

data/ Raw & processed data

notebooks/ Jupyter notebooks

reports/ Graphs, summaries

src/ Scripts

README.md  Main documentation


Future Work
- Add elevation and grid access data
- Develop predictive model for 10-year irradiance forecast
- Integrate cost analysis for solar investment



