import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Festival Finanz Dashboard", layout="wide")

# === SIDEBAR: Einstellungen ===
st.sidebar.header("Parameter & Einstellungen")

# Gästeanzahl (n)
n = st.sidebar.slider("Anzahl der Gäste (n)", min_value=50, max_value=250, value=150, step=10)

# Ticketpreis (t)
t = st.sidebar.slider("Ticketpreis (t) in €", min_value=50, max_value=150, value=70, step=1)

# Durchschnittliche Getränke pro Gast (k)
k = st.sidebar.slider("Durchschnittliche Getränke pro Gast (k)", min_value=0, max_value=20, value=5, step=1)

# Basis-Fixe Kosten (c_base) - nun ohne Gästekosten
c_base = st.sidebar.slider("Basis Fixkosten (ohne Gästekosten) in €", min_value=0, max_value=20000, value=5000, step=100)

# Fixkosten pro Gast
c_per_guest = st.sidebar.number_input("Fixkosten pro Gast in €", min_value=0, max_value=100, value=30, step=1)

# Berechnung der gesamten Fixkosten
c = c_base + (c_per_guest * n)

st.sidebar.markdown(f"**Gesamte Fixkosten: {c:.2f} €** (Basis + {c_per_guest}€ pro Gast)")
st.sidebar.markdown("---")

# === MAIN CONTENT ===
st.title("Festival Finanz Dashboard")
st.markdown("""
Diese Dashboard hilft bei der Planung des Preisgefüges für ein Festival. 
Unser Ziel ist es, möglichst günstige Getränkepreise anzubieten und trotzdem wirtschaftlich zu sein.
""")

# === Getränketabelle im Hauptbereich ===
st.header("Getränkedaten & Preiskalkulation")

col1, col2 = st.columns([2, 1])

with col1:
    # Standard-Daten: Getränk, Verkaufspreis, prozentuale Gewichtung und Herstellungs-/Einkaufskosten pro Getränk
    default_data = {
        "Getränk": ["Bier", "Wein", "Cocktail", "Spritzer", "Antialkoholisches"],
        "Preis": [5, 6, 10, 7, 4],              # Verkaufspreis
        "Gewichtung (%)": [30, 20, 15, 20, 15],  # Prozentuale Anteile
        "Kosten": [2, 3, 5, 4, 2]               # Kosten pro Getränk
    }

    # Nutze st.data_editor für eine interaktive Tabelle
    drinks_df = st.data_editor(
        pd.DataFrame(default_data),
        num_rows="dynamic",
        use_container_width=True
    )

with col2:
    # Marge-Anpassung
    st.subheader("Preisgestaltung")
    margin_reduction = st.slider(
        "Margenreduktion (%)", 
        min_value=0, 
        max_value=100, 
        value=0, 
        step=5,
        help="Reduziert die Marge (Preis - Kosten) für alle Getränke um den angegebenen Prozentsatz."
    )
    
    # Alternative Preisstrategien
    pricing_strategy = st.selectbox(
        "Preisstrategie",
        ["Benutzerdefiniert", "Cost-Plus (Minimale Marge)", "Break-Even", "Volumen-basiert"],
        help="Wähle eine alternative Preisstrategie oder definiere deine eigenen Preise in der Tabelle."
    )

# Aktualisiere Preise basierend auf der gewählten Strategie
if pricing_strategy != "Benutzerdefiniert" and not drinks_df.empty:
    original_df = drinks_df.copy()
    
    if pricing_strategy == "Cost-Plus (Minimale Marge)":
        # 10% Aufschlag auf Kosten
        drinks_df["Preis"] = drinks_df["Kosten"] * 1.1
        st.info("Cost-Plus-Strategie: 10% Aufschlag auf die Kosten")
        
    elif pricing_strategy == "Break-Even":
        # Berechne den minimalen Preis, der zum Break-Even führt
        # Die Berechnung ist komplexer, da sie vom gesamten Festival-Finanzmodell abhängt
        total_costs = c
        total_drinks = n * k
        
        # Gewichtete Kosten der Getränke
        if drinks_df["Gewichtung (%)"].sum() > 0:
            norm_weights = drinks_df["Gewichtung (%)"] / drinks_df["Gewichtung (%)"].sum()
            weighted_costs = drinks_df["Kosten"] * norm_weights * total_drinks
            
            # Minimaler Aufschlag, um Kosten zu decken
            min_markup = (total_costs - weighted_costs.sum()) / total_drinks if total_drinks > 0 else 0
            min_markup = max(0, min_markup)  # Nicht negativ
            
            # Neue Preise = Kosten + minimaler Aufschlag
            drinks_df["Preis"] = drinks_df["Kosten"] + min_markup
            
            st.info(f"Break-Even-Strategie: Minimaler Aufschlag von {min_markup:.2f}€ pro Getränk, um alle Kosten zu decken")
        
    elif pricing_strategy == "Volumen-basiert":
        # Höhere Rabatte für populärere Getränke
        max_weight = drinks_df["Gewichtung (%)"].max()
        if max_weight > 0:
            # Berechne Rabatt basierend auf relativem Gewicht
            relative_weight = drinks_df["Gewichtung (%)"] / max_weight
            # Populärere Getränke bekommen bis zu 30% Rabatt auf die Marge
            discount_factor = 1 - (relative_weight * 0.3)
            # Anwenden der Rabatte
            margin = drinks_df["Preis"] - drinks_df["Kosten"]
            discounted_margin = margin * discount_factor
            drinks_df["Preis"] = drinks_df["Kosten"] + discounted_margin
            
            st.info("Volumen-basierte Strategie: Höhere Rabatte für beliebtere Getränke")

# Wende Margenreduktion an, wenn gewählt
if margin_reduction > 0 and not drinks_df.empty:
    margin = drinks_df["Preis"] - drinks_df["Kosten"]
    reduced_margin = margin * (1 - margin_reduction / 100)
    drinks_df["Preis"] = drinks_df["Kosten"] + reduced_margin
    st.warning(f"Alle Margen wurden um {margin_reduction}% reduziert!")

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

st.markdown(f"**Durchschnittlicher Verkaufspreis:** {d_s:.2f} €")
st.markdown(f"**Durchschnittliche Getränkekosten:** {d_cost:.2f} €")
st.markdown(f"**Durchschnittliche Marge pro Getränk:** {d_s - d_cost:.2f} €")

# Marge visualisieren
if not drinks_df.empty:
    # Calculate margin as the difference between price and cost
    drinks_df["Marge"] = drinks_df["Preis"] - drinks_df["Kosten"]
    
    # Create a simple, clean bar chart
    fig_margins = go.Figure()
    
    # Add cost bars (bottom segment)
    fig_margins.add_trace(go.Bar(
        x=drinks_df["Getränk"],
        y=drinks_df["Kosten"],
        name="Kosten",
        marker_color="red"
    ))
    
    # Add margin bars on top (top segment)
    fig_margins.add_trace(go.Bar(
        x=drinks_df["Getränk"],
        y=drinks_df["Marge"],
        name="Marge",
        marker_color="green"
    ))
    
    # Update layout for a stacked bar chart
    fig_margins.update_layout(
        barmode="stack",
        title="Getränkepreise: Kosten und Marge",
        yaxis_title="Preis (€)",
        legend=dict(orientation="v", yanchor="bottom", y=0.5, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_margins, use_container_width=True)

st.markdown("---")

# === Berechnung der Einnahmen, variablen Getränkekosten und Profit ===
revenue_tickets = n * t
revenue_drinks = n * k * d_s
variable_drink_cost = n * k * d_cost
total_revenue = revenue_tickets + revenue_drinks
total_cost = c + variable_drink_cost
profit = total_revenue - total_cost

# Break-even point calculation for number of guests
if (t + k * (d_s - d_cost) - c_per_guest) <= 0:
    break_even_guests = float('inf')  # Unmöglich Break-Even zu erreichen
else:
    break_even_guests = c_base / (t + k * (d_s - d_cost) - c_per_guest)

# Format break-even point
break_even_guests_formatted = int(np.ceil(break_even_guests)) if not np.isnan(break_even_guests) and break_even_guests != float('inf') else "N/A"

# Berechne minimalen Getränkepreis für Break-Even
if n > 0 and k > 0:
    min_drink_price_for_breakeven = d_cost + (c + n * t * (-1)) / (n * k)
    min_drink_price_for_breakeven = max(d_cost, min_drink_price_for_breakeven)  # Nicht unter Kosten
else:
    min_drink_price_for_breakeven = float('inf')

# KPI Section with larger metrics
st.header("Finanzielle Kennzahlen")
col1, col2, col3 = st.columns(3)
col1.metric("Gesamteinnahmen", f"{total_revenue:.2f} €", f"{revenue_tickets:.2f} € Tickets + {revenue_drinks:.2f} € Getränke")
col2.metric("Gesamtkosten", f"{total_cost:.2f} €", f"{c:.2f} € Fix + {variable_drink_cost:.2f} € Variable")
col3.metric("Profit", f"{profit:.2f} €", f"Break-even bei {break_even_guests_formatted} Gästen")

st.markdown("---")

# === Fokus auf Getränkepreis-Optimierung ===
st.header("Getränkepreis-Optimierung")

# Erstelle zwei Spalten für die Diagramme
opt_col1, opt_col2 = st.columns(2)

with opt_col1:
    # Mittlerer Getränkepreis vs. Break-Even
    st.subheader("Mindestgetränkepreis für Break-Even")
    
    # Erzeuge einen Bereich an Getränkepreisen
    price_range = np.linspace(d_cost, d_cost * 2, 50)
    profits = [n * t + n * k * (price - d_cost) - c for price in price_range]
    
    # Erstelle das Diagramm
    fig_drink_price = go.Figure()
    
    # Gewinnlinie hinzufügen
    fig_drink_price.add_trace(go.Scatter(
        x=price_range, 
        y=profits, 
        mode='lines', 
        name='Profit',
        line=dict(color='blue', width=3)
    ))
    
    # Nulllinie hinzufügen
    fig_drink_price.add_shape(
        type="line",
        x0=d_cost,
        y0=0,
        x1=d_cost * 2,
        y1=0,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Aktuellen Punkt markieren
    fig_drink_price.add_trace(go.Scatter(
        x=[d_s],
        y=[profit],
        mode='markers',
        marker=dict(color='green', size=12),
        name='Aktueller Preis'
    ))
    
    # Minimalen Break-Even-Preis markieren
    if not np.isnan(min_drink_price_for_breakeven) and min_drink_price_for_breakeven != float('inf'):
        fig_drink_price.add_trace(go.Scatter(
            x=[min_drink_price_for_breakeven],
            y=[0],
            mode='markers',
            marker=dict(color='red', size=12, symbol='diamond'),
            name=f'Min. Break-Even: {min_drink_price_for_breakeven:.2f}€'
        ))
    
    # Layout aktualisieren
    fig_drink_price.update_layout(
        xaxis_title="Durchschnittlicher Getränkepreis (€)",
        yaxis_title="Profit (€)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_drink_price, use_container_width=True)

with opt_col2:
    # Auswirkung der Getränkepreissenkung auf Profit
    st.subheader("Einfluss der Getränkepreissenkung")
    
    # Erzeuge einen Bereich von Preissenkungen (0-50%)
    reduction_range = np.linspace(0, 50, 50)
    
    # Berechne die reduzierten Preise und den resultierenden Profit
    original_margin = d_s - d_cost
    reduced_margins = [original_margin * (1 - r/100) for r in reduction_range]
    reduced_prices = [d_cost + margin for margin in reduced_margins]
    
    profits_reduction = [n * t + n * k * margin - c for margin in reduced_margins]
    
    # Erstelle das Diagramm
    fig_reduction = go.Figure()
    
    # Zwei y-Achsen für Profit und Preis
    fig_reduction = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Profitlinie hinzufügen
    fig_reduction.add_trace(
        go.Scatter(
            x=reduction_range, 
            y=profits_reduction, 
            mode='lines', 
            name='Profit',
            line=dict(color='blue', width=3)
        ),
        secondary_y=False
    )
    
    # Preislinie hinzufügen
    fig_reduction.add_trace(
        go.Scatter(
            x=reduction_range, 
            y=reduced_prices, 
            mode='lines', 
            name='Getränkepreis',
            line=dict(color='orange', width=3)
        ),
        secondary_y=True
    )
    
    # Nulllinie für Profit hinzufügen
    fig_reduction.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=50,
        y1=0,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Aktuellen Punkt markieren
    fig_reduction.add_trace(
        go.Scatter(
            x=[margin_reduction],
            y=[profit],
            mode='markers',
            marker=dict(color='green', size=12),
            name='Aktuelle Einstellung'
        ),
        secondary_y=False
    )
    
    # Layout aktualisieren
    fig_reduction.update_layout(
        xaxis_title="Margenreduktion (%)",
        hovermode="x unified"
    )
    fig_reduction.update_yaxes(title_text="Profit (€)", secondary_y=False)
    fig_reduction.update_yaxes(title_text="Getränkepreis (€)", secondary_y=True)
    
    st.plotly_chart(fig_reduction, use_container_width=True)

# === Plot: Sankey Diagram for cash flow ===
st.header("Geldfluss")

# Prepare data for Sankey diagram
labels = ["Tickets", "Getränke", "Basis Fixkosten", "Gästekosten", "Getränkekosten", "Profit"]
source = [0, 1, 0, 0, 1, 2, 3, 4]
target = [5, 5, 2, 3, 4, 5, 5, 5]
value = [
    revenue_tickets, 
    revenue_drinks, 
    c_base, 
    c_per_guest * n, 
    variable_drink_cost, 
    profit if profit > 0 else 0, 
    profit if profit > 0 else 0,
    profit if profit > 0 else 0
]

# Handle negative profit (loss)
if profit < 0:
    source = [0, 1, 0, 0, 1, 5]
    target = [2, 4, 3, 5, 5, 2]
    value = [
        c_base - profit if c_base - profit > 0 else 0, 
        variable_drink_cost, 
        c_per_guest * n, 
        revenue_tickets, 
        revenue_drinks,
        -profit if -profit > 0 else 0
    ]

# Create the Sankey diagram
fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=["blue", "green", "red", "orange", "purple", "gold" if profit >= 0 else "red"]
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=["rgba(0,0,255,0.4)", "rgba(0,255,0,0.4)", "rgba(255,0,0,0.4)", 
               "rgba(255,165,0,0.4)", "rgba(128,0,128,0.4)", 
               "rgba(255,215,0,0.4)" if profit >= 0 else "rgba(255,0,0,0.4)",
               "rgba(255,215,0,0.4)" if profit >= 0 else "rgba(255,0,0,0.4)",
               "rgba(255,215,0,0.4)" if profit >= 0 else "rgba(255,0,0,0.4)"
              ]
    )
)])

fig_sankey.update_layout(
    title_text="Geldfluss Visualisierung",
    font_size=12
)

st.plotly_chart(fig_sankey, use_container_width=True)

# === Profit Gauge ===
st.header("Profitabilität")

# Create a gauge chart
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=profit,
    title={"text": "Profit (€)"},
    delta={"reference": 0, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
    gauge={
        "axis": {"range": [-c, c*2]},
        "bar": {"color": "darkblue" if profit >= 0 else "darkred"},
        "steps": [
            {"range": [-c, 0], "color": "lightcoral"},
            {"range": [0, c*2], "color": "lightgreen"}
        ],
        "threshold": {
            "line": {"color": "red", "width": 4},
            "thickness": 0.75,
            "value": 0
        }
    }
))

st.plotly_chart(fig_gauge, use_container_width=True)

# === Profit Berechnung ===
st.header("Berechnung")
st.markdown(f"""
Die Profit-Berechnung erfolgt nach folgender Formel:

\\[
\\text{{Profit}} = n \\times t + n \\times k \\times \\bigl(d_s - d_{{cost}}\\bigr) - (c_{{base}} + n \\times c_{{per\_guest}})
\\]

mit:
- **n** = {n} (Gästeanzahl)
- **t** = {t} € (Ticketpreis)
- **k** = {k} (durchschnittliche Getränke pro Gast)
- **d_s** = {d_s:.2f} € (durchschnittlicher Verkaufspreis der Getränke)
- **d_cost** = {d_cost:.2f} € (durchschnittliche Kosten der Getränke)
- **c_base** = {c_base} € (Basis-Fixkosten)
- **c_per_guest** = {c_per_guest} € (Fixkosten pro Gast)
- **c** = {c_base} € + {n} × {c_per_guest} € = {c} € (Gesamtfixkosten)

Die Berechnung:
- Ticket Einnahmen: {n} \\(\\times\\) {t} = {revenue_tickets:.2f} €
- Getränke Einnahmen: {n} \\(\\times\\) {k} \\(\\times\\) {d_s:.2f} = {revenue_drinks:.2f} €
- Variable Getränkekosten: {n} \\(\\times\\) {k} \\(\\times\\) {d_cost:.2f} = {variable_drink_cost:.2f} €
- Gesamteinnahmen: {revenue_tickets:.2f} € + {revenue_drinks:.2f} € = {total_revenue:.2f} €
- Gesamtkosten: {c} € + {variable_drink_cost:.2f} € = {total_cost:.2f} €
- Profit: {total_revenue:.2f} € - {total_cost:.2f} € = {profit:.2f} €

**Break-even Berechnung:**
- Break-even Gästeanzahl: {break_even_guests_formatted}
- Minimaler Getränkepreis für Break-Even: {min_drink_price_for_breakeven:.2f} € (bei aktueller Gästeanzahl)
- Bei den aktuellen Einstellungen ist ein Profit von {profit:.2f} € zu erwarten.
""")
