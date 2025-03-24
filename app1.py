import streamlit as st
import pandas as pd
from datetime import datetime

def main():
    st.title("Fertilizer Inventory Management System")
    st.write("### Manage Your Fertilizer Stock Efficiently")
    
    # Initialize session state for inventory if it doesn't exist
    if 'inventory' not in st.session_state:
        st.session_state.inventory = pd.DataFrame(
            columns=['Product Name', 'Type', 'Bags Available', 'Price per Bag (₹)', 
                    'NPK Ratio', 'Last Updated', 'Min Stock Level']
        )

    # Sidebar for adding new fertilizer products
    st.sidebar.header("Add/Update Fertilizer Product")
    with st.sidebar.form("add_product"):
        product_name = st.text_input("Product Name")
        fertilizer_type = st.selectbox(
            "Fertilizer Type",
            ["Chemical", "Organic", "Bio-fertilizer", "Water Soluble"]
        )
        npk_ratio = st.text_input("NPK Ratio (e.g., 14-14-14)")
        quantity = st.number_input("Number of Bags", min_value=0, step=1)
        price = st.number_input("Price per Bag (₹)", min_value=0.0, format="%.2f")
        min_stock = st.number_input("Minimum Stock Level (bags)", min_value=0, step=1)
        
        if st.form_submit_button("Add/Update Product"):
            new_product = pd.DataFrame({
                'Product Name': [product_name],
                'Type': [fertilizer_type],
                'Bags Available': [quantity],
                'Price per Bag (₹)': [price],
                'NPK Ratio': [npk_ratio],
                'Last Updated': [datetime.now().strftime("%Y-%m-%d %H:%M")],
                'Min Stock Level': [min_stock]
            })
            
            # Update if product exists, append if new
            if product_name in st.session_state.inventory['Product Name'].values:
                idx = st.session_state.inventory['Product Name'] == product_name
                st.session_state.inventory.loc[idx] = new_product.iloc[0]
                st.success(f"Updated {product_name} in inventory!")
            else:
                st.session_state.inventory = pd.concat([st.session_state.inventory, new_product], 
                                                     ignore_index=True)
                st.success(f"Added {product_name} to inventory!")

    # Main page display
    st.header("Current Inventory Status")
    
    # Filter for low stock items
    low_stock = st.session_state.inventory[
        st.session_state.inventory['Bags Available'] <= st.session_state.inventory['Min Stock Level']
    ]
    
    if not low_stock.empty:
        st.warning("### Low Stock Alert!")
        st.dataframe(low_stock[['Product Name', 'Bags Available', 'Min Stock Level']])
    
    # Display options
    display_option = st.radio(
        "Select Display Option",
        ["All Products", "Available Products Only", "Custom View"]
    )

    # Display full inventory
    st.subheader("Inventory Display")
    if not st.session_state.inventory.empty:
        # Add a status column
        inventory_display = st.session_state.inventory.copy()
        inventory_display['Status'] = inventory_display.apply(
            lambda x: "Available" if x['Bags Available'] > 0 else "Out of Stock", axis=1
        )

        if display_option == "Available Products Only":
            inventory_display = inventory_display[inventory_display['Bags Available'] > 0]
        elif display_option == "Custom View":
            # Let user select columns to display
            columns_to_show = st.multiselect(
                "Select columns to display",
                inventory_display.columns.tolist(),
                default=['Product Name', 'Type', 'Bags Available', 'Price per Bag (₹)']
            )
            inventory_display = inventory_display[columns_to_show]
        
        # Color code the status
        def highlight_status(row):
            if row.get('Status', '') == 'Available':
                return ['background-color: #90EE90'] * len(row)
            return ['background-color: #FFB6C1'] * len(row)
        
        st.dataframe(inventory_display.style.apply(highlight_status, axis=1))
        
        # Download inventory report
        csv = inventory_display.to_csv(index=False)
        st.download_button(
            label="Download Inventory Report",
            data=csv,
            file_name="fertilizer_inventory.csv",
            mime="text/csv"
        )
    else:
        st.info("No products in inventory. Please add products using the sidebar form.")

    # Instructions
    with st.expander("How to Use This System"):
        st.write("""
        1. **Add New Product**: 
            - Use the sidebar form to add new fertilizer products
            - Fill in all required information including NPK ratio
            - Click 'Add/Update Product' to save
            
        2. **Update Existing Product**:
            - Enter the exact product name in the sidebar form
            - Update the quantities or other details
            - Click 'Add/Update Product' to save changes
            
        3. **Monitor Stock**:
            - Red highlighted rows indicate out-of-stock items
            - Green highlighted rows indicate available items
            - Low stock alerts appear at the top when stock is below minimum level
            
        4. **Display Options**:
            - All Products: Shows complete inventory
            - Available Products Only: Shows only products in stock
            - Custom View: Select specific columns to display
            
        5. **Reports**:
            - Download the complete inventory report using the download button
            - Use this for record keeping and ordering decisions
        """)

if __name__ == "__main__":
    main()
