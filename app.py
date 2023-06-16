import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# Cargando el modelo previamente entrenado
model = load_model('Home_Price_Prediction')

# Definición de valores predeterminados para cada característica
default_values = {
    'MSSubClass': 20, 'MSZoning': 'RL', 'LotFrontage': 70, 'LotArea': 10500, 'Street': 'Pave', 
    'LotShape': 'Reg', 'LandContour': 'Lvl', 'Utilities': 'AllPub', 'LotConfig': 'Inside', 
    'LandSlope': 'Gtl', 'Neighborhood': 'NAmes', 'Condition1': 'Norm', 'Condition2': 'Norm', 
    'BldgType': '1Fam', 'HouseStyle': '1Story', 'OverallQual': 6, 'OverallCond': 5, 
    'YearBuilt': 1973, 'YearRemodAdd': 1994, 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 
    'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd', 'MasVnrType': 'None', 'MasVnrArea': 0, 
    'ExterQual': 'TA', 'ExterCond': 'TA', 'Foundation': 'PConc', 'BsmtQual': 'TA', 
    'BsmtCond': 'TA', 'BsmtExposure': 'No', 'BsmtFinType1': 'Unf', 'BsmtFinSF1': 0, 
    'BsmtFinType2': 'Unf', 'BsmtFinSF2': 0, 'BsmtUnfSF': 0, 'TotalBsmtSF': 0, 'Heating': 'GasA', 
    'HeatingQC': 'TA', 'CentralAir': 'Y', 'Electrical': 'SBrkr', '1stFlrSF': 1087, 
    '2ndFlrSF': 0, 'LowQualFinSF': 0, 'GrLivArea': 1515, 'BsmtFullBath': 0, 'BsmtHalfBath': 0, 
    'FullBath': 2, 'HalfBath': 0, 'BedroomAbvGr': 3, 'KitchenAbvGr': 1, 'KitchenQual': 'TA', 
    'TotRmsAbvGrd': 6, 'Functional': 'Typ', 'Fireplaces': 0, 'GarageType': 'Attchd', 
    'GarageYrBlt': 1980, 'GarageFinish': 'Unf', 'GarageCars': 2, 'GarageArea': 480, 
    'GarageQual': 'TA', 'GarageCond': 'TA', 'PavedDrive': 'Y', 'WoodDeckSF': 0, 
    'OpenPorchSF': 25, 'EnclosedPorch': 0, '3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 
    'MiscVal': 0, 'MoSold': 6, 'YrSold': 2008, 'SaleType': 'WD', 'SaleCondition': 'Normal'
}

# Función para hacer predicciones
def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

# Función principal de Streamlit
def run():
    st.title('Predicción de precios de casas')

    # Crear un diccionario para almacenar las entradas del usuario
    user_inputs = {}

    # Opciones para variables categóricas
    MSZoning_options = ['RL', 'RM', 'C (all)', 'FV', 'RH']
    Street_options = ['Pave', 'Grvl']

    user_inputs['MSZoning'] = st.selectbox('MSZoning', options=MSZoning_options, index=MSZoning_options.index(default_values['MSZoning']))
    user_inputs['Street'] = st.selectbox('Street', options=Street_options, index=Street_options.index(default_values['Street']))

    # Inputs para variables numéricas
    user_inputs['LotFrontage'] = st.number_input('LotFrontage', value=default_values['LotFrontage'])
    user_inputs['LotArea'] = st.number_input('LotArea', value=default_values['LotArea'])
    user_inputs['YearBuilt'] = st.number_input('YearBuilt', value=default_values['YearBuilt'])

    # Convertir las entradas del usuario a un dataframe
    user_input_df = pd.DataFrame([user_inputs])

    # Reemplazar las columnas faltantes con los valores predeterminados
    for col in default_values.keys():
        if col not in user_input_df.columns:
            user_input_df[col] = default_values[col]

    # Hacer la predicción y mostrarla
    if st.button("Predict"):
        prediction = predict(model, user_input_df)
        st.write(f"El precio predicho de la casa es: {prediction}")

if __name__ == '__main__':
    run()



