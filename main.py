import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Title
st.title("Efficient Drug Discovery using Molecular Deep Learning")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Data Upload", "Model Training", "Prediction"])

# Data Upload Page
if page == "Data Upload":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())

        if st.button("Save Data"):
            data.to_csv("drug_data.csv", index=False)
            st.success("Data saved successfully!")

# Model Training Page
elif page == "Model Training":
    st.header("Train Model")

    if st.button("Load Data"):
        data = pd.read_csv("drug_data.csv")
        st.write("Data Loaded:")
        st.write(data.head())

        # Assuming the last column is the target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = Sequential()
        model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=200, batch_size=10, validation_data=(X_test, y_test))
        model.save("drug_model.h5")
        st.success("Model trained and saved successfully!")


        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

# Prediction Page
elif page == "Prediction":
    st.header("Make Predictions")

    model = load_model("drug_model.h5")

    st.write("Enter the features of the drug for prediction (comma separated):")
    input_data = [float(x) for x in st.text_input("Features (comma separated)").split(",")]

    if st.button("Predict"):
        if len(input_data) != 4:
            st.error("Please enter exactly 4 features.")
        else:
            input_data = np.array(input_data).reshape(1, -1)

            # Debug: print input data
            st.write("Input Data:")
            st.write(input_data)

            prediction = model.predict(input_data)
            prediction = (prediction > 0.5).astype(int)

            # Debug: print raw prediction value
            st.write("Raw Prediction Value:")
            st.write(prediction)

            st.write("Prediction: ", "Drug is viable" if prediction[0][0] == 1 else "Drug is not viable")
