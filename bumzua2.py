import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Festival Finanz Dashboard", layout="wide")

# === SIDEBAR: Einstellungen ===
st.sidebar.header("Parameter & Einstellungen")

# Gästeanzahl (n)
n = st.sidebar.slider("Anzahl der Gäste (n)", min_value=0, max_value=400, value=200, step=1)

# Ticketpreis (t)
t = st.sidebar.slider("Ticketpreis (t) in €", min_value=50, max_value=150, value=100, step=1)

# Durchschnittliche Getränke pro Gast (k)
k = st.sidebar.slider("Durchschnittliche Getränke pro Gast (k)", min_value=0, max_value=100, value=3, step=1)

# Fixe Kosten (c)
c = st.sidebar.slider("Fixe Kosten (c) in €", min_value=0, max_value=20000, value=5000, step=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Getränkedaten")

# === Interaktive Getränketabelle ===
# Standard-Daten: Getränk, Verkaufspreis, prozentuale Gewichtung und Herstellungs-/Einkaufskosten pro Getränk
default_data = {
    "Getränk": ["Bier", "Wein", "Cocktail", "Spritzer", "Antialkoholisches"],
    "Preis": [5, 6, 10, 7, 4],              # Verkaufspreis
    "Gewichtung (%)": [30, 20, 15, 20, 15],  # Prozentuale Anteile
    "Kosten": [2, 3, 5, 4, 2]               # Kosten pro Getränk
}

# Nutze st.data_editor (aktuelle Streamlit-Version) für eine interaktive Tabelle
drinks_df = st.data_editor(
    pd.DataFrame(default_data),
    num_rows="dynamic",
    use_container_width=True
)

# Berechne den durchschnittlichen Verkaufspreis (d_s) und den durchschnittlichen Getränkekosten (d_cost) als gewichtete Mittelwerte
if not drinks_df.empty:
    total_weight = drinks_df["Gewichtung (%)"].sum()
    if total_weight == 0:
        d_s = 0
        d_cost = 0
    else:
        norm_weights = drinks_df["Gewichtung (%)"] / total_weight
        d_s = (drinks_df["Preis"] * norm_weights).sum()
        d_cost = (drinks_df["Kosten"] * norm_weights).sum()
else:
    d_s = 0
    d_cost = 0

st.markdown(f"**Berechneter durchschnittlicher Verkaufspreis (d_s):** {d_s:.2f} €")
st.markdown(f"**Berechnete durchschnittliche Getränkekosten (d_cost):** {d_cost:.2f} €")

# === Berechnung der Einnahmen, variablen Getränkekosten und Profit ===
revenue_tickets = n * t
revenue_drinks = n * k * d_s
variable_drink_cost = n * k * d_cost
total_revenue = revenue_tickets + revenue_drinks
total_cost = c + variable_drink_cost
profit = total_revenue - total_cost

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ticket Einnahmen", f"{revenue_tickets:.2f} €")
col2.metric("Getränke Einnahmen", f"{revenue_drinks:.2f} €")
col3.metric("Gesamteinnahmen", f"{total_revenue:.2f} €")
col4.metric("Profit", f"{profit:.2f} €")

st.markdown("---")

# === Plot 1: Konturplot: Ticketpreis vs. durchschnittlicher Verkaufspreis der Getränke und Profit ===
st.subheader("Profit-Kontur: Ticketpreis vs. Getränkepreis")
# Erzeuge ein Gitter für t (50-150) und d_s (1-20)
t_values = np.linspace(50, 150, 50)
d_values = np.linspace(1, 20, 50)
T_grid, D_grid = np.meshgrid(t_values, d_values)
# Die Profitformel: Profit = n*t + n*k*(d_s - d_cost) - c
profit_grid = n * T_grid + n * k * (D_grid - d_cost) - c

fig1 = go.Figure(data=go.Contour(
    z=profit_grid,
    x=t_values,  # Ticketpreis
    y=d_values,  # Durchschnittlicher Verkaufspreis (d_s)
    colorscale='Viridis',
    contours=dict(showlabels=True),
    colorbar=dict(title="Profit (€)")
))
# Aktuellen Punkt markieren (t, d_s)
fig1.add_trace(go.Scatter(
    x=[t],
    y=[d_s],
    mode='markers',
    marker=dict(color='red', size=12),
    name='Aktuelle Einstellung'
))
fig1.update_layout(
    title="Profit-Kontur: Ticketpreis (t) vs. Getränkepreis (d_s)",
    xaxis_title="Ticketpreis (t) [€]",
    yaxis_title="Durchschnittlicher Verkaufspreis (d_s) [€]"
)
st.plotly_chart(fig1, use_container_width=True)

# === Plot 2: Profit-Gauge ===
st.subheader("Profit Visualisierung")
fig2 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=profit,
    title={"text": "Profit"},
    gauge={
        "axis": {"range": [-10000, 100000]},
        "bar": {"color": "darkblue"},
        "steps": [
            {"range": [-10000, 0], "color": "red"},
            {"range": [0, 100000], "color": "green"}
        ],
    }
))
st.plotly_chart(fig2, use_container_width=True)

# === Weitere nützliche Informationen ===
st.subheader("Profit Berechnung")
st.markdown(f"""
Die Profit-Berechnung erfolgt nach folgender Formel:

\\[
\\text{{Profit}} = n \\times t + n \\times k \\times \\bigl(d_s - d_{{cost}}\\bigr) - c
\\]

mit:
- **n** = {n} (Gästeanzahl)
- **t** = {t} € (Ticketpreis)
- **k** = {k} (durchschnittliche Getränke pro Gast)
- **d_s** = {d_s:.2f} € (durchschnittlicher Verkaufspreis der Getränke)
- **d_cost** = {d_cost:.2f} € (durchschnittliche Kosten der Getränke)
- **c** = {c} € (fixe Kosten)

Die Berechnung:
- Ticket Einnahmen: {n} \\(\\times\\) {t} = {revenue_tickets:.2f} €
- Getränke Einnahmen: {n} \\(\\times\\) {k} \\(\\times\\) {d_s:.2f} = {revenue_drinks:.2f} €
- Variable Getränkekosten: {n} \\(\\times\\) {k} \\(\\times\\) {d_cost:.2f} = {variable_drink_cost:.2f} €
- Gesamteinnahmen: {revenue_tickets:.2f} € + {revenue_drinks:.2f} € = {total_revenue:.2f} €
- Gesamtkosten: {c} € + {variable_drink_cost:.2f} € = {total_cost:.2f} €
- Profit: {total_revenue:.2f} € - {total_cost:.2f} € = {profit:.2f} €
""")
