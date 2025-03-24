import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
from datetime import timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import qrcode
from PIL import Image
import base64
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import barcode
from barcode.writer import ImageWriter
import calendar
import matplotlib.pyplot as plt
import seaborn as sns

def highlight_status(row):
    """Highlight row based on stock status with enhanced color scheme"""
    colors = {
        'Critical': '#FF0000',  # Red
        'Low': '#FFA500',      # Orange  
        'Normal': '#90EE90',   # Light Green
        'Excess': '#006400'    # Dark Green
    }
    status = row['Stock Status']
    return [f'background-color: {colors[status]}; color: white; font-weight: bold'] * len(row)

def generate_qr_code(data):
    """Generate QR code for product/inventory data"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def send_alert_email(recipient, subject, body, attachment=None):
    """Send email alerts for low stock, expiry etc."""
    sender_email = st.secrets["email"]["sender"]
    password = st.secrets["email"]["password"]
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))
    
    if attachment:
        with open(attachment, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(attachment))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment)}"'
        msg.attach(part)
        
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, password)
        server.send_message(msg)

def generate_barcode(product_code):
    """Generate barcode for products"""
    EAN = barcode.get_barcode_class('ean13')
    ean = EAN(product_code, writer=ImageWriter())
    buffered = BytesIO()
    ean.write(buffered)
    return base64.b64encode(buffered.getvalue()).decode()

def advanced_sales_forecast(historical_data, periods=30):
    """Advanced sales forecasting using machine learning"""
    X = np.array(range(len(historical_data))).reshape(-1, 1)
    y = historical_data['Daily Sales'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_dates = pd.date_range(start=datetime.now(), periods=periods, freq='D')
    future_X = np.array(range(len(X), len(X) + periods)).reshape(-1, 1)
    forecast = model.predict(future_X)
    
    confidence = 1 - mean_squared_error(y, model.predict(X)) / np.var(y)
    
    return pd.DataFrame({
        'Date': future_dates,
        'Forecasted Sales': forecast,
        'Confidence': [confidence] * periods
    })

def export_to_tally_format(inventory_data):
    """Enhanced Tally export with additional fields"""
    tally_data = {
        "ENVELOPE": {
            "HEADER": {
                "TALLYREQUEST": "Import Data",
                "VERSION": "1.0",
                "TIMESTAMP": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "BODY": {
                "IMPORTDATA": {
                    "REQUESTDESC": {
                        "REPORTNAME": "Stock Items",
                        "STATICVARIABLES": {
                            "SVCURRENTCOMPANY": "##SVCURRENTCOMPANY",
                            "SVEXPORTFORMAT": "$$SysName:XML"
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
                "PARENT": row['Type'],
                "CATEGORY": row['Category'],
                "OPENINGBALANCE": str(row['Available Bags']),
                "OPENINGRATE": str(row['Price per Bag (₹)']),
                "OPENINGVALUE": str(row['Available Bags'] * row['Price per Bag (₹)']),
                "BATCHALLOCATIONS.LIST": {
                    "NAME": row['NPK Ratio'],
                    "BATCHNAME": row['Last Updated'],
                    "MINLEVEL": str(row['Minimum Stock Level']),
                    "MAXLEVEL": str(row['Maximum Stock Level']),
                    "GODOWNNAME": row['Storage Location'],
                    "EXPIRYDATE": row['Expiry Date'],
                    "MANUFACTURINGDATE": row['Manufacturing Date']
                },
                "GSTDETAILS.LIST": {
                    "APPLICABLEFROM": datetime.now().strftime("%Y%m%d"),
                    "CALCULATIONTYPE": "On Value",
                    "HSNCODE": row['HSN Code'],
                    "TAXABILITY": "Taxable",
                    "GSTTYPEOFSUPPLY": "Goods"
                }
            }
        }
        tally_data["ENVELOPE"]["BODY"]["IMPORTDATA"]["REQUESTDATA"]["TALLYMESSAGE"].append(stock_item)
    
    return tally_data

def generate_sales_forecast(historical_data):
    """Generate comprehensive sales forecast"""
    basic_forecast = advanced_sales_forecast(historical_data)
    
    # Add seasonality
    month = datetime.now().month
    season_factor = {
        'Spring': 1.2 if month in [3,4,5] else 1.0,
        'Summer': 0.8 if month in [6,7,8] else 1.0,
        'Fall': 1.1 if month in [9,10,11] else 1.0,
        'Winter': 0.9 if month in [12,1,2] else 1.0
    }
    
    basic_forecast['Seasonal Adjustment'] = basic_forecast['Forecasted Sales'].apply(
        lambda x: x * season_factor[calendar.month_name[month]]
    )
    
    return basic_forecast

def print_bill(invoice_data):
    """Send the invoice data to a remote printer"""
    # Here you would implement the logic to send the invoice data to a printer
    # This could involve using a cloud printing service or an API that interfaces with the printer
    # For demonstration, we will just print the invoice data to the console
    print("Printing Invoice...")
    print(invoice_data)

def main():
    st.set_page_config(layout="wide", page_title="Advanced Fertilizer Management System")
    
    # Add data persistence
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        
        # Try to load existing data
        try:
            with open('fertilizer_data.json', 'r') as f:
                data = json.load(f)
                st.session_state.inventory = pd.DataFrame.from_dict(data['inventory'])
                st.session_state.sales_history = pd.DataFrame.from_dict(data['sales'])
                st.session_state.customers = pd.DataFrame.from_dict(data['customers'])
                st.session_state.suppliers = pd.DataFrame.from_dict(data['suppliers'])
                st.session_state.data_loaded = True
        except FileNotFoundError:
            # Initialize empty DataFrames if no existing data
            st.session_state.inventory = pd.DataFrame(
                columns=[
                    'Product Name', 'Type', 'Category', 'Available Bags', 'Price per Bag (₹)', 
                    'NPK Ratio', 'Last Updated', 'Minimum Stock Level', 'Maximum Stock Level',
                    'Daily Sales', 'Storage Location', 'Manufacturer', 'Batch Number',
                    'Manufacturing Date', 'Expiry Date', 'HSN Code', 'GST Rate',
                    'Reorder Point', 'Lead Time (Days)', 'Stock Status', 'QR Code',
                    'Barcode', 'Safety Stock', 'ABC Classification'
                ]
            )
            st.session_state.sales_history = pd.DataFrame(
                columns=[
                    'Date', 'Product', 'Sales Quantity', 'Total Value', 'Customer Name',
                    'Payment Mode', 'Invoice Number', 'Salesperson', 'Region', 'Discount',
                    'Net Amount', 'GST Amount', 'Profit Margin'
                ]
            )
            st.session_state.customers = pd.DataFrame(
                columns=[
                    'Customer Name', 'Type', 'Contact Person', 'Email', 'Phone',
                    'Address', 'Credit Limit', 'Payment Terms', 'Loyalty Points',
                    'Last Purchase Date', 'Total Purchases', 'Rating'
                ]
            )
            st.session_state.suppliers = pd.DataFrame(
                columns=[
                    'Supplier Name', 'Contact Person', 'Email', 'Phone', 'Address',
                    'Products Supplied', 'Payment Terms', 'Lead Time', 'Rating',
                    'Last Order Date', 'Contract Expiry'
                ]
            )
    
    # Add auto-save functionality after any data modification
    def save_data():
        data = {
            'inventory': st.session_state.inventory.to_dict(),
            'sales': st.session_state.sales_history.to_dict(),
            'customers': st.session_state.customers.to_dict(),
            'suppliers': st.session_state.suppliers.to_dict()
        }
        with open('fertilizer_data.json', 'w') as f:
            json.dump(data, f, default=str)
    
    # Add auto-save to forms
    if st.session_state.get('form_submitted', False):
        save_data()
        st.session_state.form_submitted = False

    # Enhanced session state initialization
    if 'inventory' not in st.session_state:
        st.session_state.inventory = pd.DataFrame(
            columns=[
                'Product Name', 'Type', 'Category', 'Available Bags', 'Price per Bag (₹)', 
                'NPK Ratio', 'Last Updated', 'Minimum Stock Level', 'Maximum Stock Level',
                'Daily Sales', 'Storage Location', 'Manufacturer', 'Batch Number',
                'Manufacturing Date', 'Expiry Date', 'HSN Code', 'GST Rate',
                'Reorder Point', 'Lead Time (Days)', 'Stock Status', 'QR Code',
                'Barcode', 'Safety Stock', 'ABC Classification'
            ]
        )
    
    if 'sales_history' not in st.session_state:
        st.session_state.sales_history = pd.DataFrame(
            columns=[
                'Date', 'Product', 'Sales Quantity', 'Total Value', 'Customer Name',
                'Payment Mode', 'Invoice Number', 'Salesperson', 'Region', 'Discount',
                'Net Amount', 'GST Amount', 'Profit Margin'
            ]
        )
    
    if 'suppliers' not in st.session_state:
        st.session_state.suppliers = pd.DataFrame(
            columns=[
                'Supplier Name', 'Contact Person', 'Email', 'Phone', 'Address',
                'Products Supplied', 'Payment Terms', 'Lead Time', 'Rating',
                'Last Order Date', 'Contract Expiry'
            ]
        )
    
    if 'customers' not in st.session_state:
        st.session_state.customers = pd.DataFrame(
            columns=[
                'Customer Name', 'Type', 'Contact Person', 'Email', 'Phone',
                'Address', 'Credit Limit', 'Payment Terms', 'Loyalty Points',
                'Last Purchase Date', 'Total Purchases', 'Rating'
            ]
        )

    # Enhanced navigation with modern UI
    PAGES = {
        "📊 Dashboard": "dashboard",
        "📦 Inventory Management": "inventory", 
        "💰 Sales & Billing": "sales",
        "📈 Analytics & Forecasting": "analytics",
        "👥 Customer Management": "customers",
        "🏭 Supplier Management": "suppliers",
        "📝 Reports & Documents": "reports",
        "⚙️ Settings": "settings"
    }
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", list(PAGES.keys()), label_visibility="collapsed")

    if PAGES[page] == "dashboard":
        st.title("Dashboard")
        # Dashboard implementation
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Products", len(st.session_state.inventory))
        with col2:
            st.metric("Total Sales", len(st.session_state.sales_history))
        with col3:
            st.metric("Total Customers", len(st.session_state.customers))

    elif PAGES[page] == "inventory":
        st.title("Inventory Management")
        
        # Add new product form
        with st.form("add_product"):
            st.subheader("Add New Product")
            col1, col2 = st.columns(2)
            
            with col1:
                product_name = st.text_input("Product Name")
                product_type = st.selectbox("Type", ["Chemical", "Organic", "Bio-fertilizer", "Water Soluble"])
                category = st.text_input("Category")
                available_bags = st.number_input("Available Bags", min_value=0)
                price = st.number_input("Price per Bag (₹)", min_value=0.0)
                
            with col2:
                npk_ratio = st.text_input("NPK Ratio (e.g., 14-14-14)")
                min_stock = st.number_input("Minimum Stock Level", min_value=0)
                max_stock = st.number_input("Maximum Stock Level", min_value=0)
                location = st.text_input("Storage Location")
                manufacturer = st.text_input("Manufacturer")
            
            if st.form_submit_button("Add Product"):
                new_product = pd.DataFrame({
                    'Product Name': [product_name],
                    'Type': [product_type],
                    'Category': [category],
                    'Available Bags': [available_bags],
                    'Price per Bag (₹)': [price],
                    'NPK Ratio': [npk_ratio],
                    'Last Updated': [datetime.now().strftime("%Y-%m-%d %H:%M")],
                    'Minimum Stock Level': [min_stock],
                    'Maximum Stock Level': [max_stock],
                    'Storage Location': [location],
                    'Manufacturer': [manufacturer]
                })
                
                st.session_state.inventory = pd.concat([st.session_state.inventory, new_product], 
                                                     ignore_index=True)
                st.session_state.form_submitted = True
                st.success(f"Added {product_name} to inventory!")
        
        # Display current inventory
        st.subheader("Current Inventory")
        if not st.session_state.inventory.empty:
            st.dataframe(st.session_state.inventory)
        else:
            st.info("No products in inventory yet.")

    elif PAGES[page] == "sales":
        st.title("Sales & Billing")
        
        # Create new sale form
        with st.form("create_sale"):
            st.subheader("Create New Sale")
            col1, col2 = st.columns(2)
            
            with col1:
                customer_name = st.text_input("Customer Name")
                sale_date = st.date_input("Sale Date")
                payment_mode = st.selectbox("Payment Mode", ["Cash", "Credit", "UPI", "Bank Transfer"])
                salesperson = st.text_input("Salesperson")
                region = st.text_input("Region")
            
            with col2:
                product = st.selectbox("Select Product", 
                    options=st.session_state.inventory['Product Name'].unique() if not st.session_state.inventory.empty else [])
                quantity = st.number_input("Quantity (Bags)", min_value=1)
                discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0)
                
                if product:
                    price = st.session_state.inventory[
                        st.session_state.inventory['Product Name'] == product
                    ]['Price per Bag (₹)'].iloc[0]
                    total = price * quantity
                    discounted_total = total * (1 - discount/100)
                    st.write(f"Total Amount: ₹{discounted_total:,.2f}")
            
            if st.form_submit_button("Create Sale"):
                if not product:
                    st.error("Please select a product")
                else:
                    # Update inventory
                    idx = st.session_state.inventory['Product Name'] == product
                    if st.session_state.inventory.loc[idx, 'Available Bags'].iloc[0] >= quantity:
                        st.session_state.inventory.loc[idx, 'Available Bags'] -= quantity
                        
                        # Record sale
                        new_sale = pd.DataFrame({
                            'Date': [sale_date],
                            'Product': [product],
                            'Sales Quantity': [quantity],
                            'Total Value': [total],
                            'Customer Name': [customer_name],
                            'Payment Mode': [payment_mode],
                            'Invoice Number': [f"INV-{len(st.session_state.sales_history) + 1}"],
                            'Salesperson': [salesperson],
                            'Region': [region],
                            'Discount': [discount],
                            'Net Amount': [discounted_total],
                            'GST Amount': [discounted_total * 0.18],  # Assuming 18% GST
                            'Profit Margin': [(discounted_total - (price * 0.7 * quantity)) / discounted_total * 100]
                        })
                        
                        st.session_state.sales_history = pd.concat([st.session_state.sales_history, new_sale],
                                                                ignore_index=True)
                        st.session_state.form_submitted = True
                        st.success(f"Sale recorded successfully! Invoice: INV-{len(st.session_state.sales_history)}")
                        
                        # Print the bill
                        print_bill(new_sale.to_dict(orient='records')[0])  # Convert to dict for printing
                    else:
                        st.error(f"Insufficient stock! Available: {st.session_state.inventory.loc[idx, 'Available Bags'].iloc[0]} bags")
        
        # Display sales history
        st.subheader("Sales History")
        if not st.session_state.sales_history.empty:
            st.dataframe(st.session_state.sales_history)
            
            # Sales analytics
            col1, col2 = st.columns(2)
            with col1:
                # Replace Plotly bar chart with Matplotlib
                fig, ax = plt.subplots()
                sns.barplot(data=st.session_state.sales_history, x='Date', y='Net Amount', ax=ax)
                ax.set_title('Daily Sales')
                ax.set_xlabel('Date')
                ax.set_ylabel('Net Amount')
                st.pyplot(fig)
            
            with col2:
                # Replace Plotly pie chart with Matplotlib
                fig, ax = plt.subplots()
                sales_by_product = st.session_state.sales_history.groupby('Product')['Sales Quantity'].sum()
                ax.pie(sales_by_product, labels=sales_by_product.index, autopct='%1.1f%%')
                ax.set_title('Sales by Product')
                st.pyplot(fig)
        else:
            st.info("No sales recorded yet.")

    elif PAGES[page] == "analytics":
        st.title("Analytics & Forecasting")
        
        tab1, tab2, tab3 = st.tabs(["Sales Analytics", "Inventory Analytics", "Forecasting"])
        
        with tab1:
            st.subheader("Sales Performance Analysis")
            if not st.session_state.sales_history.empty:
                # Time period selector
                period = st.selectbox(
                    "Select Time Period",
                    ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Time"]
                )
                
                # Calculate key metrics
                total_sales = st.session_state.sales_history['Net Amount'].sum()
                avg_order_value = st.session_state.sales_history['Net Amount'].mean()
                total_orders = len(st.session_state.sales_history)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sales", f"₹{total_sales:,.2f}")
                with col2:
                    st.metric("Average Order Value", f"₹{avg_order_value:,.2f}")
                with col3:
                    st.metric("Total Orders", total_orders)
                
                # Sales trends
                fig, ax = plt.subplots()
                sns.lineplot(data=st.session_state.sales_history, x='Date', y='Net Amount', ax=ax)
                ax.set_title('Sales Trend')
                ax.set_xlabel('Date')
                ax.set_ylabel('Net Amount')
                st.pyplot(fig)
                
                # Product performance
                product_sales = st.session_state.sales_history.groupby('Product').agg({
                    'Sales Quantity': 'sum',
                    'Net Amount': 'sum',
                    'Profit Margin': 'mean'
                }).reset_index()
                
                st.subheader("Product Performance")
                st.dataframe(product_sales)
            else:
                st.info("No sales data available for analysis")
        
        with tab2:
            st.subheader("Inventory Analysis")
            if not st.session_state.inventory.empty:
                # Inventory value
                total_value = (st.session_state.inventory['Available Bags'] * 
                             st.session_state.inventory['Price per Bag (₹)']).sum()
                
                # Stock status
                st.session_state.inventory['Stock Status'] = st.session_state.inventory.apply(
                    lambda x: 'Critical' if x['Available Bags'] <= x['Minimum Stock Level'] 
                    else 'Low' if x['Available Bags'] <= x['Minimum Stock Level'] * 1.5
                    else 'Normal' if x['Available Bags'] <= x['Maximum Stock Level']
                    else 'Excess',
                    axis=1
                )
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Inventory Value", f"₹{total_value:,.2f}")
                with col2:
                    st.metric("Total Products", len(st.session_state.inventory))
                
                # Stock status breakdown
                fig, ax = plt.subplots()
                sns.countplot(data=st.session_state.inventory, x='Stock Status', ax=ax)
                ax.set_title('Stock Status Distribution')
                st.pyplot(fig)
                
                # Low stock alerts
                low_stock = st.session_state.inventory[
                    st.session_state.inventory['Stock Status'].isin(['Critical', 'Low'])
                ]
                if not low_stock.empty:
                    st.warning("Low Stock Alerts")
                    st.dataframe(low_stock[['Product Name', 'Available Bags', 'Minimum Stock Level', 'Stock Status']])
            else:
                st.info("No inventory data available for analysis")
        
        with tab3:
            st.subheader("Sales Forecasting")
            if not st.session_state.sales_history.empty:
                forecast_days = st.slider("Forecast Days", 7, 90, 30)
                
                # Generate forecast
                forecast_df = advanced_sales_forecast(st.session_state.sales_history, forecast_days)
                
                # Display forecast
                fig, ax = plt.subplots()
                sns.lineplot(data=forecast_df, x='Date', y='Forecasted Sales', ax=ax)
                ax.set_title('Sales Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales')
                st.pyplot(fig)
                
                # Forecast metrics
                st.subheader("Forecast Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Average Forecasted Sales",
                        f"{forecast_df['Forecasted Sales'].mean():.2f} units"
                    )
                with col2:
                    st.metric(
                        "Forecast Confidence",
                        f"{forecast_df['Confidence'].iloc[0]*100:.1f}%"
                    )
                
                st.dataframe(forecast_df)
            else:
                st.info("Insufficient data for forecasting")

    elif PAGES[page] == "customers":
        st.title("Customer Management")
        
        tab1, tab2 = st.tabs(["Add/Edit Customer", "Customer List"])
        
        with tab1:
            with st.form("add_customer"):
                st.subheader("Add New Customer")
                col1, col2 = st.columns(2)
                
                with col1:
                    customer_name = st.text_input("Customer Name")
                    customer_type = st.selectbox("Customer Type", ["Retail", "Wholesale", "Distributor"])
                    contact_person = st.text_input("Contact Person")
                    email = st.text_input("Email")
                    phone = st.text_input("Phone")
                    
                with col2:
                    address = st.text_area("Address")
                    credit_limit = st.number_input("Credit Limit (₹)", min_value=0.0)
                    payment_terms = st.selectbox("Payment Terms", ["Immediate", "Net 15", "Net 30", "Net 45"])
                    loyalty_points = st.number_input("Initial Loyalty Points", min_value=0)
                    rating = st.slider("Customer Rating", 1, 5, 3)
                
                if st.form_submit_button("Add Customer"):
                    new_customer = pd.DataFrame({
                        'Customer Name': [customer_name],
                        'Type': [customer_type],
                        'Contact Person': [contact_person],
                        'Email': [email],
                        'Phone': [phone],
                        'Address': [address],
                        'Credit Limit': [credit_limit],
                        'Payment Terms': [payment_terms],
                        'Loyalty Points': [loyalty_points],
                        'Last Purchase Date': [None],
                        'Total Purchases': [0],
                        'Rating': [rating]
                    })
                    
                    if customer_name in st.session_state.customers['Customer Name'].values:
                        idx = st.session_state.customers['Customer Name'] == customer_name
                        st.session_state.customers.loc[idx] = new_customer.iloc[0]
                        st.session_state.form_submitted = True
                        st.success(f"Updated customer: {customer_name}")
                    else:
                        st.session_state.customers = pd.concat([st.session_state.customers, new_customer],
                                                           ignore_index=True)
                        st.session_state.form_submitted = True
                        st.success(f"Added new customer: {customer_name}")
        
        with tab2:
            st.subheader("Customer Database")
            if not st.session_state.customers.empty:
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    filter_type = st.multiselect(
                        "Filter by Type",
                        options=st.session_state.customers['Type'].unique()
                    )
                with col2:
                    search_term = st.text_input("Search Customers", "")
                
                filtered_customers = st.session_state.customers
                if filter_type:
                    filtered_customers = filtered_customers[filtered_customers['Type'].isin(filter_type)]
                if search_term:
                    filtered_customers = filtered_customers[
                        filtered_customers['Customer Name'].str.contains(search_term, case=False) |
                        filtered_customers['Contact Person'].str.contains(search_term, case=False)
                    ]
                
                # Display customers
                st.dataframe(filtered_customers)
                
                # Customer analytics
                st.subheader("Customer Analytics")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots()
                    sns.countplot(data=filtered_customers, x='Type', ax=ax)
                    ax.set_title('Customer Distribution by Type')
                    st.pyplot(fig)
                
                with col2:
                    if not st.session_state.sales_history.empty:
                        customer_sales = st.session_state.sales_history.groupby('Customer Name')['Net Amount'].sum().reset_index()
                        fig, ax = plt.subplots()
                        sns.barplot(data=customer_sales.sort_values('Net Amount', ascending=False).head(10), x='Customer Name', y='Net Amount', ax=ax)
                        ax.set_title('Top 10 Customers by Sales')
                        st.pyplot(fig)
                
                # Export options
                if st.button("Export Customer List"):
                    csv = filtered_customers.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="customer_list.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No customers in database yet.")

    elif PAGES[page] == "suppliers":
        st.title("Supplier Management")
        
        tab1, tab2 = st.tabs(["Add/Edit Supplier", "Supplier List"])
        
        with tab1:
            with st.form("add_supplier"):
                st.subheader("Add New Supplier")
                col1, col2 = st.columns(2)
                
                with col1:
                    supplier_name = st.text_input("Supplier Name")
                    contact_person = st.text_input("Contact Person")
                    email = st.text_input("Email")
                    phone = st.text_input("Phone")
                    address = st.text_area("Address")
                    
                with col2:
                    products = st.multiselect(
                        "Products Supplied",
                        options=st.session_state.inventory['Product Name'].unique() if not st.session_state.inventory.empty else []
                    )
                    payment_terms = st.selectbox("Payment Terms", ["Advance", "Immediate", "Net 30", "Net 60"])
                    lead_time = st.number_input("Lead Time (Days)", min_value=1)
                    rating = st.slider("Supplier Rating", 1, 5, 3)
                    contract_expiry = st.date_input("Contract Expiry Date")
                
                if st.form_submit_button("Add Supplier"):
                    new_supplier = pd.DataFrame({
                        'Supplier Name': [supplier_name],
                        'Contact Person': [contact_person],
                        'Email': [email],
                        'Phone': [phone],
                        'Address': [address],
                        'Products Supplied': [', '.join(products)],
                        'Payment Terms': [payment_terms],
                        'Lead Time': [lead_time],
                        'Rating': [rating],
                        'Last Order Date': [None],
                        'Contract Expiry': [contract_expiry]
                    })
                    
                    if supplier_name in st.session_state.suppliers['Supplier Name'].values:
                        idx = st.session_state.suppliers['Supplier Name'] == supplier_name
                        st.session_state.suppliers.loc[idx] = new_supplier.iloc[0]
                        st.session_state.form_submitted = True
                        st.success(f"Updated supplier: {supplier_name}")
                    else:
                        st.session_state.suppliers = pd.concat([st.session_state.suppliers, new_supplier],
                                                           ignore_index=True)
                        st.session_state.form_submitted = True
                        st.success(f"Added new supplier: {supplier_name}")
        
        with tab2:
            st.subheader("Supplier Database")
            if not st.session_state.suppliers.empty:
                # Filter and search
                col1, col2 = st.columns(2)
                with col1:
                    rating_filter = st.multiselect(
                        "Filter by Rating",
                        options=sorted(st.session_state.suppliers['Rating'].unique())
                    )
                with col2:
                    search_term = st.text_input("Search Suppliers", "")
                
                filtered_suppliers = st.session_state.suppliers
                if rating_filter:
                    filtered_suppliers = filtered_suppliers[filtered_suppliers['Rating'].isin(rating_filter)]
                if search_term:
                    filtered_suppliers = filtered_suppliers[
                        filtered_suppliers['Supplier Name'].str.contains(search_term, case=False) |
                        filtered_suppliers['Contact Person'].str.contains(search_term, case=False)
                    ]
                
                # Display suppliers
                st.dataframe(filtered_suppliers)
                
                # Supplier analytics
                st.subheader("Supplier Analytics")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots()
                    sns.countplot(data=filtered_suppliers, x='Rating', ax=ax)
                    ax.set_title('Supplier Ratings Distribution')
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots()
                    sns.boxplot(data=filtered_suppliers, y='Lead Time', ax=ax)
                    ax.set_title('Lead Time Distribution')
                    st.pyplot(fig)
                
                # Contract expiry alerts
                upcoming_expiry = st.session_state.suppliers[
                    pd.to_datetime(st.session_state.suppliers['Contract Expiry']) <= 
                    (datetime.now() + timedelta(days=30))
                ]
                if not upcoming_expiry.empty:
                    st.warning("Contracts Expiring Soon")
                    st.dataframe(upcoming_expiry[['Supplier Name', 'Contract Expiry']])
                
                # Export options
                if st.button("Export Supplier List"):
                    csv = filtered_suppliers.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="supplier_list.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No suppliers in database yet.")

    elif PAGES[page] == "reports":
        st.title("Reports & Documents")
        
        tab1, tab2 = st.tabs(["Generate Reports", "Export Data"])
        
        with tab1:
            st.subheader("Report Generation")
            report_type = st.selectbox(
                "Select Report Type",
                ["Sales Report", "Inventory Report", "Customer Report", "Supplier Report"]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")
            
            if st.button("Generate Report"):
                if report_type == "Sales Report" and not st.session_state.sales_history.empty:
                    # Filter sales data by date
                    mask = (pd.to_datetime(st.session_state.sales_history['Date']) >= start_date) & \
                          (pd.to_datetime(st.session_state.sales_history['Date']) <= end_date)
                    filtered_sales = st.session_state.sales_history[mask]
                    
                    if not filtered_sales.empty:
                        st.write("### Sales Summary")
                        total_sales = filtered_sales['Net Amount'].sum()
                        total_quantity = filtered_sales['Sales Quantity'].sum()
                        avg_margin = filtered_sales['Profit Margin'].mean()
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Sales", f"₹{total_sales:,.2f}")
                        col2.metric("Total Quantity", f"{total_quantity:,}")
                        col3.metric("Average Margin", f"{avg_margin:.1f}%")
                        
                        st.write("### Sales by Product")
                        product_summary = filtered_sales.groupby('Product').agg({
                            'Sales Quantity': 'sum',
                            'Net Amount': 'sum',
                            'Profit Margin': 'mean'
                        }).reset_index()
                        st.dataframe(product_summary)
                        
                        # Download option
                        csv = product_summary.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Report",
                            data=csv,
                            file_name=f"sales_report_{start_date}_to_{end_date}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No sales data for selected period")
                
                elif report_type == "Inventory Report" and not st.session_state.inventory.empty:
                    st.write("### Inventory Status")
                    inventory_summary = st.session_state.inventory.copy()
                    inventory_summary['Stock Value'] = inventory_summary['Available Bags'] * \
                                                     inventory_summary['Price per Bag (₹)']
                    
                    total_value = inventory_summary['Stock Value'].sum()
                    st.metric("Total Inventory Value", f"₹{total_value:,.2f}")
                    
                    st.dataframe(inventory_summary)
                    
                    csv = inventory_summary.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Report",
                        data=csv,
                        file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                elif report_type == "Customer Report" and not st.session_state.customers.empty:
                    st.write("### Customer Analysis")
                    customer_summary = st.session_state.customers.copy()
                    
                    if not st.session_state.sales_history.empty:
                        sales_by_customer = st.session_state.sales_history.groupby('Customer Name')['Net Amount'].sum()
                        customer_summary['Total Sales'] = customer_summary['Customer Name'].map(sales_by_customer)
                    
                    st.dataframe(customer_summary)
                    
                    csv = customer_summary.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Report",
                        data=csv,
                        file_name=f"customer_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                elif report_type == "Supplier Report" and not st.session_state.suppliers.empty:
                    st.write("### Supplier Analysis")
                    supplier_summary = st.session_state.suppliers.copy()
                    
                    st.dataframe(supplier_summary)
                    
                    csv = supplier_summary.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Report",
                        data=csv,
                        file_name=f"supplier_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with tab2:
            st.subheader("Export Data")
            export_type = st.selectbox(
                "Select Data to Export",
                ["All Data", "Inventory", "Sales", "Customers", "Suppliers"]
            )
            
            if st.button("Export"):
                if export_type == "All Data":
                    data = {
                        'inventory': st.session_state.inventory.to_dict(),
                        'sales': st.session_state.sales_history.to_dict(),
                        'customers': st.session_state.customers.to_dict(),
                        'suppliers': st.session_state.suppliers.to_dict()
                    }
                    json_str = json.dumps(data, indent=2, default=str)
                    st.download_button(
                        label="📥 Download All Data",
                        data=json_str,
                        file_name="fertilizer_management_system_data.json",
                        mime="application/json"
                    )
                else:
                    data_map = {
                        "Inventory": st.session_state.inventory,
                        "Sales": st.session_state.sales_history,
                        "Customers": st.session_state.customers,
                        "Suppliers": st.session_state.suppliers
                    }
                    if data_map[export_type].empty:
                        st.info(f"No {export_type.lower()} data to export")
                    else:
                        csv = data_map[export_type].to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {export_type} Data",
                            data=csv,
                            file_name=f"{export_type.lower()}_data.csv",
                            mime="text/csv"
                        )

    elif PAGES[page] == "settings":
        st.title("Settings")
        
        tab1, tab2, tab3 = st.tabs(["System Settings", "Email Settings", "Data Management"])
        
        with tab1:
            st.subheader("System Configuration")
            
            # General Settings
            with st.form("general_settings"):
                st.write("General Settings")
                col1, col2 = st.columns(2)
                
                with col1:
                    company_name = st.text_input("Company Name", "My Fertilizer Company")
                    currency = st.selectbox("Currency", ["₹ (INR)", "$ (USD)", "€ (EUR)"])
                    language = st.selectbox("Language", ["English", "Hindi"])
                    
                with col2:
                    gst_rate = st.number_input("Default GST Rate (%)", value=18.0)
                    date_format = st.selectbox("Date Format", ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"])
                    theme = st.selectbox("Theme", ["Light", "Dark"])
                
                if st.form_submit_button("Save General Settings"):
                    st.success("Settings saved successfully!")
            
            # Notification Settings
            with st.form("notification_settings"):
                st.write("Notification Settings")
                
                low_stock_alert = st.number_input("Low Stock Alert Threshold (%)", value=20)
                enable_email = st.checkbox("Enable Email Notifications")
                enable_mobile = st.checkbox("Enable Mobile Notifications")
                
                if st.form_submit_button("Save Notification Settings"):
                    st.success("Notification settings updated!")
        
        with tab2:
            st.subheader("Email Configuration")
            
            with st.form("email_settings"):
                st.write("Email Server Settings")
                
                smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", value=587)
                email_address = st.text_input("Email Address")
                email_password = st.text_input("Email Password", type="password")
                
                st.write("Notification Recipients")
                stock_alerts_to = st.text_input("Stock Alerts Recipients (comma-separated)")
                sales_reports_to = st.text_input("Sales Reports Recipients (comma-separated)")
                
                if st.form_submit_button("Save Email Settings"):
                    # Here you would typically save these to a secure configuration
                    st.success("Email settings updated successfully!")
        
        with tab3:
            st.subheader("Data Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Backup Data")
                if st.button("Create Backup"):
                    data = {
                        'inventory': st.session_state.inventory.to_dict(),
                        'sales': st.session_state.sales_history.to_dict(),
                        'customers': st.session_state.customers.to_dict(),
                        'suppliers': st.session_state.suppliers.to_dict()
                    }
                    json_str = json.dumps(data, indent=2, default=str)
                    st.download_button(
                        label="📥 Download Backup",
                        data=json_str,
                        file_name=f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                st.write("Restore Data")
                uploaded_file = st.file_uploader("Upload Backup File", type=['json'])
                if uploaded_file is not None:
                    if st.button("Restore from Backup"):
                        try:
                            data = json.loads(uploaded_file.getvalue())
                            st.session_state.inventory = pd.DataFrame.from_dict(data['inventory'])
                            st.session_state.sales_history = pd.DataFrame.from_dict(data['sales'])
                            st.session_state.customers = pd.DataFrame.from_dict(data['customers'])
                            st.session_state.suppliers = pd.DataFrame.from_dict(data['suppliers'])
                            st.success("Data restored successfully!")
                        except Exception as e:
                            st.error(f"Error restoring data: {str(e)}")
            
            # Data Reset Options
            st.write("---")
            st.write("⚠️ Danger Zone")
            
            if st.button("Reset All Data", type="secondary"):
                confirm = st.checkbox("I understand this will delete all data permanently")
                if confirm:
                    if st.button("Confirm Reset", type="primary"):
                        # Reset all session state data
                        for key in ['inventory', 'sales_history', 'customers', 'suppliers']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.success("All data has been reset!")
                        st.experimental_rerun()

if __name__ == "__main__":
    main()
