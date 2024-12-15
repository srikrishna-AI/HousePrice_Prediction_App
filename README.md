# House Price Prediction Web App

## Overview
This is a web application that predicts house prices based on user input features. The app allows users to input details such as house attributes and receive a predicted price using a trained linear regression model.

The application is built using **Streamlit**, **pandas**, **scikit-learn**, and other libraries.

---

## Features
- **Dataset Display**: View the first few rows of the dataset.
- **Data Preprocessing**: Automatically handles missing values and encodes categorical data.
- **Model Training**: Train a linear regression model directly from the web app.
- **Model Persistence**: Save the trained model to a file for future use.
- **Prediction**: Predict house prices based on user-provided inputs.

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

Replace `app.py` with the name of your Python script file if it's different.

---

## File Structure
- **`HousePricePrediction.csv`**: Input dataset for training and testing the model.
- **`app.py`**: The main Streamlit application script.
- **`house_price_model.pkl`**: File to save/load the trained model.

---

## Usage
1. Launch the web app using the `streamlit run` command.
2. View the dataset by checking the **Show Dataset** checkbox.
3. Train and save the model using the **Train and Save Model** button.
4. Enter the details of the house features in the input fields.
5. Click **Predict House Price** to see the predicted house price.

---

## Notes
- Ensure the dataset file (`HousePricePrediction.csv`) is in the same directory as the script or provide the correct file path in the code.
- The target column in the dataset should be named `SalePrice`. Update the code if your dataset uses a different column name for the target.

---

## Dependencies
- Streamlit
- Pandas
- Scikit-learn
- Numpy
- Pickle (for model serialization)

---

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code with attribution.
