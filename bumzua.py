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
k = st.sidebar.slider("Durchschnittliche Getränke pro Gast (k)", min_value=0, max_value=10, value=3, step=1)

# Kosten (c)
c = st.sidebar.slider("Kosten (c) in €", min_value=0, max_value=10000, value=5500, step=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Getränkedaten")

# === Interaktive Getränketabelle ===
# Standard-Daten für Getränke: Getränk, Preis, prozentuale Gewichtung
default_data = {
    "Getränk": ["Bier", "Wein", "Cocktail", "Spritzer", "Antialkoholisches"],
    "Preis": [5, 6, 10, 7, 4],
    "Gewichtung (%)": [30, 20, 15, 20, 15]
}
# Mit st.experimental_data_editor (oder st.data_editor bei neueren Streamlit-Versionen) kann die Tabelle interaktiv bearbeitet werden
drinks_df = st.sidebar.data_editor(pd.DataFrame(default_data), num_rows="dynamic", use_container_width=True)

# Berechne den durchschnittlichen Getränkepreis (d) als gewichteten Mittelwert
if not drinks_df.empty:
    total_weight = drinks_df["Gewichtung (%)"].sum()
    if total_weight == 0:
        d = 0
    else:
        # Normiere die Gewichtungen, falls sie nicht 100% ergeben
        norm_weights = drinks_df["Gewichtung (%)"] / total_weight
        d = (drinks_df["Preis"] * norm_weights).sum()
else:
    d = 0

st.markdown(f"**Berechneter durchschnittlicher Getränkepreis (d):** {d:.2f} €")

# === Berechnung der Einnahmen und Profit ===
revenue_tickets = n * t
revenue_drinks = n * k * d
total_revenue = revenue_tickets + revenue_drinks
profit = total_revenue - c

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ticket Einnahmen", f"{revenue_tickets:.2f} €")
col2.metric("Getränke Einnahmen", f"{revenue_drinks:.2f} €")
col3.metric("Gesamteinnahmen", f"{total_revenue:.2f} €")
col4.metric("Profit", f"{profit:.2f} €")

st.markdown("---")

# === Plot 1: Konturplot: Ticketpreis vs. Getränkepreis und Profit ===
st.subheader("Profit-Kontur: Ticketpreis vs. Getränkepreis")
# Erzeuge ein Gitter für t und d. Wir nutzen hier einen festen Bereich für t (50-150) und für d (1-20)
t_values = np.linspace(50, 150, 50)
d_values = np.linspace(1, 20, 50)
T_grid, D_grid = np.meshgrid(t_values, d_values)
# Profit-Formel: p = n*(t + k*d) - c
profit_grid = n * (T_grid + k * D_grid) - c

fig1 = go.Figure(data=go.Contour(
    z=profit_grid,
    x=t_values,  # Ticketpreis
    y=d_values,  # Getränkepreis
    colorscale='Viridis',
    contours=dict(showlabels=True),
    colorbar=dict(title="Profit (€)")
))
# Markiere den aktuellen Punkt (t, d)
fig1.add_trace(go.Scatter(
    x=[t],
    y=[d],
    mode='markers',
    marker=dict(color='red', size=12),
    name='Aktuelle Einstellung'
))
fig1.update_layout(
    title="Profit-Kontur: Ticketpreis (t) vs. Getränkepreis (d)",
    xaxis_title="Ticketpreis (t) [€]",
    yaxis_title="Durchschnittlicher Getränkepreis (d) [€]"
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
\\text{{Profit}} = n \\times \\bigl(t + k \\times d\\bigr) - c
\\]

mit:
- **n** = {n} (Gästeanzahl)
- **t** = {t} € (Ticketpreis)
- **k** = {k} (durchschnittliche Getränke pro Gast)
- **d** = {d:.2f} € (durchschnittlicher Getränkepreis, berechnet aus der Tabelle)
- **c** = {c} € (Kosten)

Das ergibt:
\\[
\\text{{Profit}} = {n} \\times \\bigl({t} + {k} \\times {d:.2f}\\bigr) - {c} = {profit:.2f} \\;€
\\]
""")
