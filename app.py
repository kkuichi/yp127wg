# Importovanie potrebných knižníc
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Načítanie uloženého modelu, prahu a názvov premenných
model = joblib.load("bankruptcy_model.pkl")
best_threshold = joblib.load("best_threshold.pkl")
feature_names = joblib.load("feature_names.pkl")

# Načítanie testovacích dát
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

# Názov aplikácie
st.title("What-if analýza rizika bankrotu")

# Krátky popis aplikácie
st.write("""
Táto aplikácia umožňuje analyzovať, ako sa zmení pravdepodobnosť bankrotu podniku
pri zmene jedného alebo viacerých finančných ukazovateľov.
""")

# Tabuľka s interpretáciou pravdepodobnosti bankrotu
st.markdown("""
### Interpretácia pravdepodobnosti bankrotu

| Pravdepodobnosť | Úroveň rizika | Význam |
|---|---|---|
| 0% – 30% | Nízke riziko | Podnik má iba menšie finančné problémy |
| 30% – 60% | Stredné riziko | Podnik môže mať finančné ťažkosti |
| 60% – 80% | Vysoké riziko | Finančná situácia podniku je vážna |
| 80% – 100% | Veľmi vysoké riziko | Podnik je silne ohrozený bankrotom |
""")

# Výber podniku
st.subheader("Výber podniku")

all_indices = y_test.index.tolist()

selected_index = st.selectbox(
    "Vyberte podnik z testovacej množiny:",
    all_indices
)

# Získanie dát vybraného podniku
company = X_test.loc[[selected_index]].copy()

# Výpočet pôvodnej pravdepodobnosti bankrotu
original_probability = model.predict_proba(company)[:, 1][0]

# Nastavenie rozhodovacieho prahu
st.subheader("Nastavenie rozhodovacieho prahu")


threshold_choice = st.radio(
    "Aký rozhodovací prah chcete použiť?",
    [
        "Použiť optimalizovaný threshold modelu",
        "Zadať vlastný threshold manuálne"
    ]
)

# Použitie optimalizovaného alebo vlastného prahu
if threshold_choice == "Použiť optimalizovaný threshold modelu":

    selected_threshold = float(best_threshold)

else:

    selected_threshold_percent = st.slider(
        "Akú pravdepodobnosť bankrotu považujete za dostatočne vysokú na klasifikáciu podniku ako bankrot?",
        min_value=0,
        max_value=100,
        value=int(float(best_threshold) * 100),
        step=1
    )

    selected_threshold = selected_threshold_percent / 100


# Určenie pôvodnej klasifikácie
original_prediction = int(
    original_probability >= selected_threshold
)

original_label = (
    "Bankrot"
    if original_prediction == 1
    else "Nebankrot"
)

# Zobrazenie pôvodného stavu podniku
st.subheader("Pôvodný stav podniku")

st.write(
    f"Pravdepodobnosť bankrotu: **{original_probability:.4f}**"
)


st.write(
    f"Klasifikácia modelu: **{original_label}**"
)


# Sekcia pre vytvorenie what-if scenára
st.subheader("What-if scenár")

selected_features = st.multiselect(
    "Vyberte finančné ukazovatele, ktoré chcete zmeniť:",
    feature_names
)

# Kópia dát pre simulovaný scenár
scenario_company = company.copy()

# Slovník na uloženie zmien
changes = {}

# Aplikovanie zmien na vybrané ukazovatele
for feature in selected_features:

    change_percent = st.slider(
        f"Zmena pre ukazovateľ: {feature}",
        min_value=-100,
        max_value=100,
        value=0,
        step=5
    )

    changes[feature] = change_percent

    scenario_company[feature] = (
        scenario_company[feature] * (1 + change_percent / 100)
    )


# Výpočet novej pravdepodobnosti bankrotu
new_probability = model.predict_proba(
    scenario_company
)[:, 1][0]

new_prediction = int(
    new_probability >= selected_threshold
)

new_label = (
    "Bankrot"
    if new_prediction == 1
    else "Nebankrot"
)

# Výpočet rozdielu pravdepodobností
difference = (
    new_probability - original_probability
)

# Zobrazenie výsledkov simulovaného scenára
st.subheader("Výsledok simulovaného scenára")

st.write(
    f"Pôvodná pravdepodobnosť bankrotu: **{original_probability:.4f}**"
)

st.write(
    f"Nová pravdepodobnosť bankrotu: **{new_probability:.4f}**"
)

st.write(
    f"Zmena pravdepodobnosti: **{difference:.4f}**"
)

st.write(
    f"Nová klasifikácia modelu: **{new_label}**"
)

# Zobrazenie tabuľky zmenených ukazovateľov
if selected_features:

    changes_table = pd.DataFrame({
        "Ukazovateľ": list(changes.keys()),
        "Zmena (%)": list(changes.values())
    })

    st.subheader("Zmenené ukazovatele")

    st.dataframe(changes_table)

else:

    st.info(
        "Vyberte aspoň jeden ukazovateľ pre vytvorenie scenára."
    )


# Vytvorenie tabuľky s porovnaním výsledkov
results = pd.DataFrame({
    "Stav": [
        "Pôvodný stav",
        "Simulovaný scenár"
    ],
    "Pravdepodobnosť bankrotu": [
        original_probability,
        new_probability
    ],
    "Predikcia": [
        original_label,
        new_label
    ]
})

# Zobrazenie porovnania výsledkov
st.subheader("Porovnanie výsledkov")

st.dataframe(results)


# Vizualizácia zmeny pravdepodobnosti
st.subheader("Vizualizácia zmeny pravdepodobnosti")

fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(
    data=results,
    x="Stav",
    y="Pravdepodobnosť bankrotu",
    ax=ax
)

ax.set_ylim(0, 1)

ax.set_ylabel(
    "Pravdepodobnosť bankrotu"
)

ax.set_xlabel("")

ax.set_title(
    "Porovnanie pôvodného a simulovaného scenára"
)

st.pyplot(fig)