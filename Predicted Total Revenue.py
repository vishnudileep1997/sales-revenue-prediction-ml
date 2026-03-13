import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- LOAD FILES ----------------
model = joblib.load("revenue_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Amazon Revenue Predictor", layout="centered")

st.title("📦 Amazon Sales Revenue Prediction")
st.write("Enter product and order details to predict revenue.")

# ---------------- INPUTS ----------------

product_category = st.selectbox(
    "Product Category",
    label_encoders["product_category"].classes_
)

price = st.number_input("Price", min_value=0.0,max_value=500.0)

discount_percent = st.number_input(
    "Discount Percent",
    min_value=0.0,
    max_value=100.0
)

quantity_sold = st.number_input(
    "Quantity Sold",
    min_value=1,
    max_value=5
)

customer_region = st.selectbox(
    "Customer Region",
    label_encoders["customer_region"].classes_
)

payment_method = st.selectbox(
    "Payment Method",
    label_encoders["payment_method"].classes_
)

rating = st.number_input(
    "Rating",
    min_value=0.0,
    max_value=5.0
)

review_count = st.number_input(
    "Review Count",
    min_value=0
)

# ---------------- PREDICTION ----------------

if st.button("Predict Revenue"):

    input_data = pd.DataFrame({
        'product_category': [product_category],
        'price': [price],
        'discount_percent': [discount_percent],
        'quantity_sold': [quantity_sold],
        'customer_region': [customer_region],
        'payment_method': [payment_method],
        'rating': [rating],
        'review_count': [review_count]
    })

    # Encode categorical columns
    for col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Ensure same column order as training
    input_data = input_data[model_columns]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # ---------------- KPI METRICS ----------------
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Predicted Revenue", f"${prediction[0]:,.2f}")
    col2.metric("🛒 Quantity Sold", quantity_sold)
    col3.metric("⭐ Rating", rating)

    # ---------------- GRAPH 1 ----------------
    st.markdown("---")
    st.subheader("📊 Revenue Breakdown")

    calculated_revenue = price * quantity_sold
    discount_amount = calculated_revenue * (discount_percent / 100)

    labels = ['Original Revenue', 'Discount Amount', 'Predicted Revenue']
    values = [calculated_revenue, discount_amount, prediction[0]]

    fig1, ax1 = plt.subplots()
    ax1.bar(labels, values)
    ax1.set_ylabel("Amount ($)")

    st.pyplot(fig1)

    # ---------------- GRAPH 2 ----------------
    st.markdown("---")
    st.subheader("💸 Price vs Discount")

    discounted_price = price * (1 - discount_percent / 100)

    fig2, ax2 = plt.subplots()
    ax2.bar(
        ['Original Price', 'Discounted Price'],
        [price, discounted_price]
    )

    ax2.set_ylabel("Price ($)")

    st.pyplot(fig2)

    # ---------------- GRAPH 3 ----------------
    st.markdown("---")
    st.subheader("⭐ Product Popularity")

    fig3, ax3 = plt.subplots()
    ax3.bar(
        ['Rating', 'Review Count'],
        [rating, review_count]
    )

    st.pyplot(fig3)