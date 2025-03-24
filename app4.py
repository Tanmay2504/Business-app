import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

def highlight_status(row):
    """Highlight row based on stock status"""
    return ['background-color: #90EE90' if val == 'Available' else 'background-color: #FFB6C1' 
            for val in row]

def export_to_tally_format(inventory_data):
    """Convert inventory data to Tally-compatible XML/JSON format"""
    tally_data = {
        "ENVELOPE": {
            "HEADER": {
                "TALLYREQUEST": "Import Data"
            },
            "BODY": {
                "IMPORTDATA": {
                    "REQUESTDESC": {
                        "REPORTNAME": "Stock Items",
                        "STATICVARIABLES": {
                            "SVCURRENTCOMPANY": "##SVCURRENTCOMPANY"
                        }
                    },
                    "REQUESTDATA": {
                        "TALLYMESSAGE": []
                    }
                }
            }
        }
    }
    
    for _, row in inventory_data.iterrows():
        stock_item = {
            "STOCKITEM": {
                "NAME": row['Product Name'],
                "PARENT": row['Type'],  # Category/Group
                "OPENINGBALANCE": str(row['Available Bags']),
                "OPENINGRATE": str(row['Price per Bag (₹)']),
                "OPENINGVALUE": str(row['Available Bags'] * row['Price per Bag (₹)']),
                "BATCHALLOCATIONS.LIST": {
                    "NAME": row['NPK Ratio'],
                    "BATCHNAME": row['Last Updated'],
                    "MINLEVEL": str(row['Minimum Stock Level'])
                }
            }
        }
        tally_data["ENVELOPE"]["BODY"]["IMPORTDATA"]["REQUESTDATA"]["TALLYMESSAGE"].append(stock_item)
    
    return tally_data

def generate_sales_forecast(historical_data):
    """Generate simple sales forecast based on historical data"""
    # Placeholder for more sophisticated forecasting logic
    avg_daily_sales = historical_data['Daily Sales'].mean()
    forecast_days = 30
    forecast = pd.DataFrame({
        'Date': [datetime.now() + timedelta(days=x) for x in range(forecast_days)],
        'Forecasted Sales': [avg_daily_sales * (1 + 0.01 * x) for x in range(forecast_days)]
    })
    return forecast

def main():
    st.set_page_config(layout="wide")
    st.title("Fertilizer Inventory Management System")
    st.write("### Efficiently Manage Your Fertilizer Stock")
    
    # Initialize session states
    if 'inventory' not in st.session_state:
        st.session_state.inventory = pd.DataFrame(
            columns=['Product Name', 'Type', 'Available Bags', 'Price per Bag (₹)', 
                    'NPK Ratio', 'Last Updated', 'Minimum Stock Level', 'Daily Sales']
        )
    
    if 'sales_history' not in st.session_state:
        st.session_state.sales_history = pd.DataFrame(
            columns=['Date', 'Product', 'Sales Quantity', 'Total Value']
        )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
        ["Inventory Management", "Sales Analysis", "Forecast", "Reports"])

    if page == "Inventory Management":
        # Existing inventory management code
        st.sidebar.header("Add/Update Fertilizer Product")
        with st.sidebar.form("add_product"):
            product_name = st.text_input("Product Name")
            fertilizer_type = st.selectbox(
                "Fertilizer Type",
                ["Chemical", "Organic", "Bio-fertilizer", "Water Soluble"]
            )
            npk_ratio = st.text_input("NPK Ratio (e.g. 14-14-14)")
            quantity = st.number_input("Number of Bags", min_value=0, step=1)
            price = st.number_input("Price per Bag (₹)", min_value=0.0, format="%.2f")
            min_stock = st.number_input("Minimum Stock Level (Bags)", min_value=0, step=1)
            avg_daily_sales = st.number_input("Average Daily Sales", min_value=0.0, step=0.1)
            
            if st.form_submit_button("Add/Update Product"):
                new_product = pd.DataFrame({
                    'Product Name': [product_name],
                    'Type': [fertilizer_type],
                    'Available Bags': [quantity],
                    'Price per Bag (₹)': [price],
                    'NPK Ratio': [npk_ratio],
                    'Last Updated': [datetime.now().strftime("%Y-%m-%d %H:%M")],
                    'Minimum Stock Level': [min_stock],
                    'Daily Sales': [avg_daily_sales]
                })
                
                if product_name in st.session_state.inventory['Product Name'].values:
                    idx = st.session_state.inventory['Product Name'] == product_name
                    st.session_state.inventory.loc[idx] = new_product.iloc[0]
                    st.success(f"{product_name} updated in inventory!")
                else:
                    st.session_state.inventory = pd.concat([st.session_state.inventory, new_product], 
                                                         ignore_index=True)
                    st.success(f"{product_name} added to inventory!")

        # Main inventory display with enhanced features
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Current Inventory Status")
            
            # Filter options
            filter_type = st.multiselect(
                "Filter Fertilizer Types",
                options=st.session_state.inventory['Type'].unique()
            )
            
            filtered_inventory = st.session_state.inventory
            if filter_type:
                filtered_inventory = filtered_inventory[filtered_inventory['Type'].isin(filter_type)]
            
            # Display inventory with enhanced styling
            if not filtered_inventory.empty:
                inventory_display = filtered_inventory.copy()
                inventory_display['Status'] = inventory_display.apply(
                    lambda x: "Available" if x['Available Bags'] > 0 else "Out of Stock", axis=1
                )
                inventory_display['Stock Value (₹)'] = inventory_display['Available Bags'] * inventory_display['Price per Bag (₹)']
                
                st.dataframe(inventory_display.style.apply(highlight_status, axis=1))
            
        with col2:
            # Stock value visualization
            if not st.session_state.inventory.empty:
                fig = px.pie(
                    st.session_state.inventory,
                    values='Available Bags',
                    names='Product Name',
                    title='Stock Distribution'
                )
                st.plotly_chart(fig)

    elif page == "Sales Analysis":
        st.header("Sales Analysis")
        
        # Add sales entry form
        with st.form("add_sale"):
            sale_date = st.date_input("Sale Date")
            sale_product = st.selectbox("Product", options=st.session_state.inventory['Product Name'])
            sale_quantity = st.number_input("Sale Quantity", min_value=1)
            
            if st.form_submit_button("Record Sale"):
                product_price = st.session_state.inventory[
                    st.session_state.inventory['Product Name'] == sale_product
                ]['Price per Bag (₹)'].iloc[0]
                
                new_sale = pd.DataFrame({
                    'Date': [sale_date],
                    'Product': [sale_product],
                    'Sales Quantity': [sale_quantity],
                    'Total Value': [sale_quantity * product_price]
                })
                
                st.session_state.sales_history = pd.concat([st.session_state.sales_history, new_sale],
                                                         ignore_index=True)
                st.success("Sale recorded successfully!")

        # Sales analysis visualizations
        if not st.session_state.sales_history.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    st.session_state.sales_history,
                    x='Date',
                    y='Total Value',
                    title='Daily Sales Value'
                )
                st.plotly_chart(fig)
            
            with col2:
                fig = px.line(
                    st.session_state.sales_history.groupby('Product')['Sales Quantity'].sum().reset_index(),
                    x='Product',
                    y='Sales Quantity',
                    title='Total Sales by Product'
                )
                st.plotly_chart(fig)

    elif page == "Forecast":
        st.header("Sales Forecast")
        
        if not st.session_state.sales_history.empty:
            forecast = generate_sales_forecast(st.session_state.inventory)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast['Date'],
                y=forecast['Forecasted Sales'],
                name='Forecast'
            ))
            st.plotly_chart(fig)
            
            st.write("### Forecasted Stock Requirements")
            st.dataframe(forecast)

    elif page == "Reports":
        st.header("Reports and Exports")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Regular CSV export
            csv = st.session_state.inventory.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Report",
                data=csv,
                file_name="fertilizer_inventory.csv",
                mime="text/csv"
            )
            
        with export_col2:
            # Tally-compatible export
            tally_data = export_to_tally_format(st.session_state.inventory)
            tally_json = json.dumps(tally_data, indent=2)
            st.download_button(
                label="📑 Download Tally-Compatible Data",
                data=tally_json,
                file_name="tally_import.json",
                mime="application/json"
            )

        # Additional reports
        st.subheader("Analysis Reports")
        
        if not st.session_state.inventory.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Total Inventory Value")
                total_value = (st.session_state.inventory['Available Bags'] * 
                             st.session_state.inventory['Price per Bag (₹)']).sum()
                st.metric("Total Value", f"₹{total_value:,.2f}")
            
            with col2:
                st.write("### Low Stock Products")
                low_stock = st.session_state.inventory[
                    st.session_state.inventory['Available Bags'] <= st.session_state.inventory['Minimum Stock Level']
                ]
                st.dataframe(low_stock[['Product Name', 'Available Bags', 'Minimum Stock Level']])

    # Enhanced instructions
    with st.expander("How to Use This System"):
        st.markdown("""
        ### 1. Inventory Management
        - Add new products and update existing ones
        - Monitor stock levels and prices
        - Filter by different categories
        
        ### 2. Sales Analysis
        - Record daily sales
        - View sales trends
        - Analyze product-wise performance
        
        ### 3. Forecast
        - View future sales predictions
        - Plan stock requirements
        
        ### 4. Reports
        - Export data in various formats
        - View detailed analysis reports
        - Use Tally integration
        """)

if __name__ == "__main__":
    main()
