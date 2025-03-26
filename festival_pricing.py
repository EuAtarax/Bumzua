import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Festival Pricing Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up tabs for different dashboard sections
tab1, tab2, tab3 = st.tabs(["Grunddaten & Preise", "Break-Even Analyse", "Simulation"])

# Sidebar for all inputs
with st.sidebar:
    st.title("Festival Parameter")
    
    # Guest parameters
    st.header("Gäste & Ticketing")
    n = st.slider("Gästeanzahl", 50, 250, 150, 10)
    t = st.slider("Ticketpreis (€)", 50, 150, 70, 5)
    
    # Cost parameters
    st.header("Kostenstruktur")
    c_base = st.number_input("Basis-Fixkosten (€)", 0, 50000, 10000, 500)
    c_per_guest = st.number_input("Kosten pro Gast (€)", 0, 100, 30, 5)
    
    # Drinks parameters
    st.header("Getränke")
    k = st.slider("Ø Getränke pro Gast", 0, 30, 5, 1)
    
    # Calculate total fixed costs
    total_fixed_costs = c_base + (n * c_per_guest)
    st.metric("Gesamte Fixkosten", f"{total_fixed_costs:.0f} €")

# --- Tab 1: Basic Data & Pricing ---
with tab1:
    st.header("Getränkepreise & Mengenplanung")
    
    # Introduction
    st.markdown("""
    Unser Ziel ist es, faire Getränkepreise anzubieten und trotzdem die Kosten zu decken. 
    In dieser Ansicht können Sie die Getränkepreise und Mengen anpassen.
    """)
    
    # Drinks data editor
    st.subheader("Getränkekalkulation")
    
    # Initialize session state for drinks data if not exists
    if 'drinks_data' not in st.session_state:
        st.session_state.drinks_data = pd.DataFrame({
            "Getränk": ["Bier", "Wein", "Cocktail", "Spritzer", "Antialkoholisches"],
            "Kosten (€)": [2.0, 3.0, 5.0, 4.0, 2.0],
            "Verkaufspreis (€)": [5.0, 6.0, 10.0, 7.0, 4.0],
            "Anteil (%)": [30, 20, 15, 20, 15]
        })
    
    # Create two columns for the data editor and pricing controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Data editor for drinks
        edited_df = st.data_editor(
            st.session_state.drinks_data,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Getränk": st.column_config.TextColumn("Getränk"),
                "Kosten (€)": st.column_config.NumberColumn(
                    "Kosten (€)",
                    min_value=0.0,
                    format="%.2f €"
                ),
                "Verkaufspreis (€)": st.column_config.NumberColumn(
                    "Verkaufspreis (€)",
                    min_value=0.0,
                    format="%.2f €"
                ),
                "Anteil (%)": st.column_config.NumberColumn(
                    "Anteil (%)",
                    min_value=0,
                    max_value=100,
                    format="%d %%"
                )
            },
            key="drinks_editor"
        )
        
        # Update session state
        st.session_state.drinks_data = edited_df
    
    with col2:
        st.subheader("Preisstrategie")
        
        # Pricing strategy selection
        pricing_strategy = st.radio(
            "Strategie wählen",
            ["Individuell", "Minimale Marge", "Break-Even", "Volumen-Optimiert"]
        )
        
        # Margin adjustment slider
        margin_adjustment = st.slider(
            "Marge anpassen",
            -50, 50, 0, 5,
            help="Prozentuelle Anpassung der Margen"
        )
        
        # Apply pricing changes button
        if st.button("Preise anwenden"):
            df = st.session_state.drinks_data.copy()
            
            if pricing_strategy == "Minimale Marge":
                # Add 10% to costs
                df["Verkaufspreis (€)"] = df["Kosten (€)"] * 1.1
                st.success("Minimale Marge: 10% auf Kosten")
                
            elif pricing_strategy == "Break-Even":
                # Calculate min price to break even
                if df["Anteil (%)"].sum() > 0:
                    # Normalize weights
                    weights = df["Anteil (%)"] / df["Anteil (%)"].sum()
                    
                    # Calculate weighted costs
                    total_drinks = n * k
                    weighted_costs = df["Kosten (€)"] * weights * total_drinks
                    
                    # Calculate how much revenue we need from drinks
                    total_ticket_revenue = n * t
                    required_from_drinks = total_fixed_costs - total_ticket_revenue
                    
                    # Calculate min markup per drink
                    min_markup = max(0, required_from_drinks / total_drinks)
                    
                    # Apply new prices
                    df["Verkaufspreis (€)"] = df["Kosten (€)"] + min_markup
                    st.success(f"Break-Even: Mindestaufschlag von {min_markup:.2f}€ pro Getränk")
                
            elif pricing_strategy == "Volumen-Optimiert":
                # Reduce margins more for popular drinks
                max_weight = df["Anteil (%)"].max()
                if max_weight > 0:
                    # Calculate relative weights
                    relative_weight = df["Anteil (%)"] / max_weight
                    
                    # Calculate discounts (higher volume = higher discount)
                    discounts = relative_weight * 0.2  # Up to 20% discount
                    
                    # Calculate margins and apply discounted margins
                    margins = df["Verkaufspreis (€)"] - df["Kosten (€)"]
                    df["Verkaufspreis (€)"] = df["Kosten (€)"] + margins * (1 - discounts)
                    st.success("Volumenstrategie: Beliebte Getränke günstiger")
            
            # Apply margin adjustment if needed
            if margin_adjustment != 0:
                margins = df["Verkaufspreis (€)"] - df["Kosten (€)"]
                df["Verkaufspreis (€)"] = df["Kosten (€)"] + margins * (1 + margin_adjustment/100)
                
            # Update session state
            st.session_state.drinks_data = df
    
    # Calculate average prices and costs based on weights
    drinks_df = st.session_state.drinks_data.copy()
    
    if not drinks_df.empty and drinks_df["Anteil (%)"].sum() > 0:
        # Normalize weights
        weights = drinks_df["Anteil (%)"] / drinks_df["Anteil (%)"].sum()
        
        # Calculate weighted average price and cost
        avg_price = (drinks_df["Verkaufspreis (€)"] * weights).sum()
        avg_cost = (drinks_df["Kosten (€)"] * weights).sum()
        avg_margin = avg_price - avg_cost
        
        # Calculate expected drink revenue
        drink_revenue = n * k * avg_price
        drink_cost = n * k * avg_cost
        ticket_revenue = n * t
        
        # Calculate profit
        profit = ticket_revenue + drink_revenue - total_fixed_costs - drink_cost
        
        # Display basic metrics
        st.subheader("Kerndaten")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ø Getränkepreis", f"{avg_price:.2f} €")
        col2.metric("Ø Getränkekosten", f"{avg_cost:.2f} €")
        col3.metric("Ø Marge pro Getränk", f"{avg_margin:.2f} €")
        col4.metric("Profit", f"{profit:.0f} €", f"{profit/n:.1f}€ pro Gast")
    
    # Create a simple bar chart showing cost and price using pandas
    st.subheader("Getränkepreise im Überblick")
    
    # Calculate margins for visualization
    drinks_df["Marge"] = drinks_df["Verkaufspreis (€)"] - drinks_df["Kosten (€)"]
    
    # Prepare data for stacked bar chart
    chart_data = pd.DataFrame({
        'Getränk': drinks_df["Getränk"],
        'Kosten': drinks_df["Kosten (€)"],
        'Marge': drinks_df["Marge"]
    })
    
    # Set the index to drink names
    chart_data = chart_data.set_index('Getränk')
    
    # Create figure and axes
    fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
    
    # Create stacked bar chart
    chart_data.plot(kind='bar', stacked=True, ax=ax, color=['#e74c3c', '#2ecc71'])
    
    # Add labels and title
    plt.ylabel('Preis (€)')
    plt.title('Getränkepreise: Kosten und Marge')
    plt.legend(loc='upper right')
    plt.xticks(rotation=0)  # Horizontal labels
    plt.tight_layout()
    
    # Display the chart
    st.pyplot(fig)
    
    # Display expected drink volume
    st.subheader("Erwartete Getränkemenge")
    
    # Calculate expected number of each drink
    if not drinks_df.empty and drinks_df["Anteil (%)"].sum() > 0:
        # Normalize weights again
        weights = drinks_df["Anteil (%)"] / drinks_df["Anteil (%)"].sum()
        
        # Calculate total drinks and volume for each type
        total_drinks = n * k
        drinks_df["Menge"] = weights * total_drinks
        
        # Create a pandas horizontal bar chart
        volume_data = pd.DataFrame({
            'Getränk': drinks_df["Getränk"],
            'Menge': drinks_df["Menge"].round(0)
        })
        
        # Set Getränk as index
        volume_data = volume_data.set_index('Getränk')
        
        # Create figure and axes
        fig2, ax2 = plt.figure(figsize=(10, 4)), plt.gca()
        
        # Create horizontal bar chart
        volume_data.plot(kind='barh', ax=ax2, color='#3498db')
        
        # Add labels and title
        plt.xlabel('Anzahl')
        plt.title('Erwartete Getränkemenge')
        
        # Add value labels
        for i, v in enumerate(volume_data['Menge']):
            ax2.text(v + 5, i, f'{int(v)}', va='center')
        
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig2)

# --- Tab 2: Break-Even Analysis ---
with tab2:
    st.header("Break-Even Analyse")
    
    # Introduction
    st.markdown("""
    Die Break-Even Analyse zeigt, welche Parameter nötig sind, um die Kosten zu decken.
    Hier können Sie berechnen, wie viele Gäste oder welche Getränkepreise Sie benötigen.
    """)
    
    # Get data for calculations
    drinks_df = st.session_state.drinks_data.copy()
    
    if not drinks_df.empty and drinks_df["Anteil (%)"].sum() > 0:
        # Normalize weights
        weights = drinks_df["Anteil (%)"] / drinks_df["Anteil (%)"].sum()
        
        # Calculate weighted average price and cost
        avg_price = (drinks_df["Verkaufspreis (€)"] * weights).sum()
        avg_cost = (drinks_df["Kosten (€)"] * weights).sum()
        
        # Calculate break-even point for number of guests
        ticket_margin = t - c_per_guest  # Margin per guest from tickets
        drink_margin = k * (avg_price - avg_cost)  # Margin per guest from drinks
        
        if ticket_margin + drink_margin > 0:
            break_even_guests = c_base / (ticket_margin + drink_margin)
            be_guests_formatted = round(break_even_guests)
        else:
            break_even_guests = float('inf')
            be_guests_formatted = "∞"
        
        # Calculate minimum drink price for break-even at current guest count
        if n > 0 and k > 0:
            # How much we need to cover with drinks
            to_cover = total_fixed_costs - (n * ticket_margin)
            
            # Minimum price needed per drink
            min_price_per_drink = avg_cost + (to_cover / (n * k))
            min_price_formatted = max(avg_cost, min_price_per_drink)
        else:
            min_price_per_drink = float('inf')
            min_price_formatted = "∞"
        
        # Display key break-even metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Break-Even Gästeanzahl", 
                f"{be_guests_formatted:,}", 
                f"{n - be_guests_formatted:+,} zur aktuellen Planung"
            )
            
            # Guest break-even chart
            st.subheader("Profit nach Gästeanzahl")
            
            # Generate range of guest counts
            guest_range = np.arange(50, 251, 10)
            
            # Calculate profits for each guest count
            profits = []
            for n_i in guest_range:
                fixed_costs_i = c_base + (n_i * c_per_guest)
                revenue_i = n_i * t + n_i * k * avg_price
                costs_i = fixed_costs_i + n_i * k * avg_cost
                profit_i = revenue_i - costs_i
                profits.append(profit_i)
            
            # Create line chart
            fig_guests = go.Figure()
            
            # Add profit line
            fig_guests.add_trace(go.Scatter(
                x=guest_range,
                y=profits,
                mode='lines',
                name='Profit',
                line=dict(color='blue', width=3)
            ))
            
            # Add break-even point
            if break_even_guests >= 50 and break_even_guests <= 250:
                fig_guests.add_trace(go.Scatter(
                    x=[break_even_guests],
                    y=[0],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name=f'Break-Even: {be_guests_formatted} Gäste'
                ))
            
            # Add zero line
            fig_guests.add_shape(
                type="line",
                x0=50,
                y0=0,
                x1=250,
                y1=0,
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add current point
            fig_guests.add_trace(go.Scatter(
                x=[n],
                y=[profit],
                mode='markers',
                marker=dict(color='green', size=10),
                name=f'Aktuell: {n} Gäste'
            ))
            
            # Update layout
            fig_guests.update_layout(
                xaxis_title="Anzahl Gäste",
                yaxis_title="Profit (€)",
                height=400,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig_guests, use_container_width=True)
        
        with col2:
            st.metric(
                "Minimum Getränkepreis für Break-Even", 
                f"{min_price_formatted:.2f} €", 
                f"{avg_price - min_price_formatted:.2f} € zur aktuellen Kalkulation"
            )
            
            # Drink price break-even chart
            st.subheader("Profit nach Getränkepreis")
            
            # Generate range of prices
            price_range = np.linspace(avg_cost, avg_cost * 2.5, 50)
            
            # Calculate profits for each price
            price_profits = []
            for price in price_range:
                drink_revenue_i = n * k * price
                drink_cost_i = n * k * avg_cost
                profit_i = ticket_revenue + drink_revenue_i - total_fixed_costs - drink_cost_i
                price_profits.append(profit_i)
            
            # Create line chart
            fig_prices = go.Figure()
            
            # Add profit line
            fig_prices.add_trace(go.Scatter(
                x=price_range,
                y=price_profits,
                mode='lines',
                name='Profit',
                line=dict(color='blue', width=3)
            ))
            
            # Add break-even point
            if min_price_per_drink >= avg_cost and min_price_per_drink <= avg_cost * 2.5:
                fig_prices.add_trace(go.Scatter(
                    x=[min_price_per_drink],
                    y=[0],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name=f'Break-Even: {min_price_per_drink:.2f}€'
                ))
            
            # Add zero line
            fig_prices.add_shape(
                type="line",
                x0=avg_cost,
                y0=0,
                x1=avg_cost * 2.5,
                y1=0,
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add current point
            fig_prices.add_trace(go.Scatter(
                x=[avg_price],
                y=[profit],
                mode='markers',
                marker=dict(color='green', size=10),
                name=f'Aktuell: {avg_price:.2f}€'
            ))
            
            # Update layout
            fig_prices.update_layout(
                xaxis_title="Durchschnittlicher Getränkepreis (€)",
                yaxis_title="Profit (€)",
                height=400,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig_prices, use_container_width=True)
        
        # Cash flow diagram
        st.subheader("Finanzfluss-Diagramm")
        
        # Prepare data for Sankey diagram
        ticket_revenue = n * t
        drink_revenue = n * k * avg_price
        drink_cost = n * k * avg_cost
        guest_costs = n * c_per_guest
        
        # Create lists for source, target and values
        labels = ["Tickets", "Getränke", "Basis-Fixkosten", "Gästekosten", "Getränkekosten", "Profit"]
        
        if profit >= 0:
            # Profit case
            source = [0, 1, 0, 0, 1, 2, 3, 4]
            target = [5, 5, 2, 3, 4, 5, 5, 5]
            values = [
                ticket_revenue, 
                drink_revenue, 
                c_base, 
                guest_costs, 
                drink_cost, 
                profit, 
                profit, 
                profit
            ]
            colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#2c3e50"]
        else:
            # Loss case
            source = [0, 1, 5, 0, 1]
            target = [2, 4, 2, 3, 5]
            values = [
                c_base - profit if c_base - profit > 0 else 0,
                drink_cost,
                -profit if -profit > 0 else 0,
                guest_costs,
                ticket_revenue + drink_revenue - drink_cost
            ]
            colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#c0392b"]
        
        # Create Sankey diagram
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=colors
            ),
            link=dict(
                source=source,
                target=target,
                value=values
            )
        )])
        
        fig_sankey.update_layout(
            height=400,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig_sankey, use_container_width=True)

# --- Tab 3: Scenario Simulation ---
with tab3:
    st.header("Szenario-Simulation")
    
    # Introduction
    st.markdown("""
    Hier können Sie verschiedene Szenarien für Ihr Festival simulieren und vergleichen.
    Testen Sie, wie sich verschiedene Faktoren auf Ihr finanzielles Ergebnis auswirken.
    """)
    
    # Get current data
    drinks_df = st.session_state.drinks_data.copy()
    
    if not drinks_df.empty and drinks_df["Anteil (%)"].sum() > 0:
        # Normalize weights
        weights = drinks_df["Anteil (%)"] / drinks_df["Anteil (%)"].sum()
        
        # Calculate weighted average price and cost
        avg_price = (drinks_df["Verkaufspreis (€)"] * weights).sum()
        avg_cost = (drinks_df["Kosten (€)"] * weights).sum()
        
        # Create simulation controls in 3 columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Gäste & Tickets")
            guest_range = st.slider("Gästeanzahl (±)", -50, 50, 0, 10)
            ticket_price_change = st.slider("Ticketpreis (±%)", -30, 30, 0, 5)
        
        with col2:
            st.subheader("Getränke")
            drinks_per_guest_change = st.slider("Getränke pro Gast (±%)", -30, 30, 0, 5)
            drink_price_change = st.slider("Getränkepreise (±%)", -30, 30, 0, 5)
        
        with col3:
            st.subheader("Kosten")
            fixed_costs_change = st.slider("Fixkosten (±%)", -30, 30, 0, 5)
            margin_scenario = st.radio("Preisstrategie", ["Standard", "Minimal", "Premium"])
        
        # Calculate values for the current scenario (base)
        n_current = n
        t_current = t
        k_current = k
        avg_price_current = avg_price
        c_base_current = c_base
        c_per_guest_current = c_per_guest
        
        # Calculate the modified scenario
        n_modified = n + guest_range
        t_modified = t * (1 + ticket_price_change/100)
        k_modified = k * (1 + drinks_per_guest_change/100)
        
        # Adjust drink prices based on margin scenario
        if margin_scenario == "Minimal":
            avg_price_modified = avg_cost * 1.1  # 10% margin
        elif margin_scenario == "Premium":
            avg_price_modified = avg_cost * 1.5  # 50% margin
        else:
            avg_price_modified = avg_price * (1 + drink_price_change/100)
        
        c_base_modified = c_base * (1 + fixed_costs_change/100)
        c_per_guest_modified = c_per_guest
        
        # Calculate financials for both scenarios
        # Current scenario
        total_fixed_costs_current = c_base_current + (n_current * c_per_guest_current)
        ticket_revenue_current = n_current * t_current
        drink_revenue_current = n_current * k_current * avg_price_current
        drink_cost_current = n_current * k_current * avg_cost
        total_revenue_current = ticket_revenue_current + drink_revenue_current
        total_cost_current = total_fixed_costs_current + drink_cost_current
        profit_current = total_revenue_current - total_cost_current
        
        # Modified scenario
        total_fixed_costs_modified = c_base_modified + (n_modified * c_per_guest_modified)
        ticket_revenue_modified = n_modified * t_modified
        drink_revenue_modified = n_modified * k_modified * avg_price_modified
        drink_cost_modified = n_modified * k_modified * avg_cost
        total_revenue_modified = ticket_revenue_modified + drink_revenue_modified
        total_cost_modified = total_fixed_costs_modified + drink_cost_modified
        profit_modified = total_revenue_modified - total_cost_modified
        
        # Display comparison
        st.subheader("Szenario-Vergleich")
        
        # Prepare comparison data
        comparison_data = {
            "Metrik": [
                "Gästeanzahl", 
                "Ticketpreis (€)", 
                "Getränke pro Gast",
                "Ø Getränkepreis (€)",
                "Total Fixkosten (€)",
                "Ticket-Einnahmen (€)",
                "Getränke-Einnahmen (€)",
                "Getränke-Kosten (€)",
                "Gesamteinnahmen (€)",
                "Gesamtkosten (€)",
                "Profit (€)",
                "Profit pro Gast (€)"
            ],
            "Aktuell": [
                n_current,
                round(t_current, 2),
                round(k_current, 1),
                round(avg_price_current, 2),
                round(total_fixed_costs_current, 0),
                round(ticket_revenue_current, 0),
                round(drink_revenue_current, 0),
                round(drink_cost_current, 0),
                round(total_revenue_current, 0),
                round(total_cost_current, 0),
                round(profit_current, 0),
                round(profit_current / n_current if n_current > 0 else 0, 2)
            ],
            "Simulation": [
                n_modified,
                round(t_modified, 2),
                round(k_modified, 1),
                round(avg_price_modified, 2),
                round(total_fixed_costs_modified, 0),
                round(ticket_revenue_modified, 0),
                round(drink_revenue_modified, 0),
                round(drink_cost_modified, 0),
                round(total_revenue_modified, 0),
                round(total_cost_modified, 0),
                round(profit_modified, 0),
                round(profit_modified / n_modified if n_modified > 0 else 0, 2)
            ]
        }
        
        # Calculate differences
        comparison_data["Differenz"] = [
            comparison_data["Simulation"][i] - comparison_data["Aktuell"][i] 
            for i in range(len(comparison_data["Metrik"]))
        ]
        
        # Calculate percentage changes
        comparison_data["Änderung (%)"] = [
            round((comparison_data["Simulation"][i] / comparison_data["Aktuell"][i] - 1) * 100, 1) 
            if comparison_data["Aktuell"][i] != 0 else 0
            for i in range(len(comparison_data["Metrik"]))
        ]
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Format the DataFrame for display
        st.dataframe(
            comparison_df,
            use_container_width=True,
            column_config={
                "Metrik": st.column_config.TextColumn("Metrik"),
                "Aktuell": st.column_config.NumberColumn("Aktuell"),
                "Simulation": st.column_config.NumberColumn("Simulation"),
                "Differenz": st.column_config.NumberColumn("Differenz", format="%.1f"),
                "Änderung (%)": st.column_config.NumberColumn("Änderung (%)", format="%.1f%%")
            }
        )
        
        # Visual comparison of profits using pandas bar chart
        st.subheader("Profitvergleich")
        
        # Prepare data for grouped bar chart
        categories = ["Ticket-Einnahmen", "Getränke-Einnahmen", "Fixkosten", "Getränke-Kosten", "Profit"]
        current_values = [ticket_revenue_current, drink_revenue_current, -total_fixed_costs_current, -drink_cost_current, profit_current]
        modified_values = [ticket_revenue_modified, drink_revenue_modified, -total_fixed_costs_modified, -drink_cost_modified, profit_modified]
        
        # Create DataFrame for comparison chart
        comparison_chart_data = pd.DataFrame({
            'Kategorie': categories,
            'Aktuell': current_values,
            'Simulation': modified_values
        }).set_index('Kategorie')
        
        # Create figure and axes
        fig_compare, ax_compare = plt.figure(figsize=(10, 6)), plt.gca()
        
        # Create grouped bar chart
        comparison_chart_data.plot(kind='bar', ax=ax_compare, color=['#3498db', '#e74c3c'])
        
        # Add labels and title
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xlabel('Kategorie')
        plt.ylabel('Betrag (€)')
        plt.title('Profitvergleich: Aktuell vs. Simulation')
        plt.legend(loc='upper right')
        
        # Rotate x labels for better readability
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig_compare)
        
        # Summary
        st.subheader("Ergebnis")
        
        # Profit change
        profit_change = profit_modified - profit_current
        profit_change_percent = (profit_modified / profit_current - 1) * 100 if profit_current != 0 else 0
        
        if profit_change > 0:
            st.success(f"Die Simulation führt zu einem um {profit_change:.0f}€ ({profit_change_percent:.1f}%) höheren Profit!")
        elif profit_change < 0:
            st.error(f"Die Simulation führt zu einem um {-profit_change:.0f}€ ({-profit_change_percent:.1f}%) niedrigeren Profit!")
        else:
            st.info("Die Simulation hat keine signifikante Auswirkung auf den Profit.")
        
        # Break-even analysis for modified scenario
        ticket_margin_modified = t_modified - c_per_guest_modified
        drink_margin_modified = k_modified * (avg_price_modified - avg_cost)
        
        if ticket_margin_modified + drink_margin_modified > 0:
            break_even_guests_modified = c_base_modified / (ticket_margin_modified + drink_margin_modified)
            be_guests_modified_formatted = round(break_even_guests_modified)
            
            st.info(f"Break-Even bei {be_guests_modified_formatted} Gästen (aktuell: {be_guests_formatted} Gäste)")
        else:
            st.warning("Mit diesen Einstellungen kann kein Break-Even erreicht werden!") 