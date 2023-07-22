import streamlit as st
import pandas as pd
import pip
pip.main(["install","openpyxl"])
import plotly.express as px
from PIL import Image
import requests
import os
import numpy as np
import statsmodels.api as sm

st.set_page_config(page_title="Aplicacion", page_icon="🌍", layout="wide")
st.markdown("##")

container = st.container()
col1, col2 = st.columns(2)


@st.cache_resource
def load_data(file):
    """
    Load data from a file (CSV or Excel).

    Parameters:
        file (File): The file to load.

    Returns:
        DataFrame: The loaded data.
    """
    file_extension = file.name.split(".")[-1].lower()  # Convertir la extensión del archivo a minúsculas
    if file_extension in ["xls", "xlsx"]:
        data = pd.read_excel(file)
    else:
        st.warning("Formato de archivo no soportado. Cargue un archivo Excel.")
        return None
    return data


def select_columns(df):
    st.write("### Select Columns")
    all_columns = df.columns.tolist()
    options_key = "_".join(all_columns)
    selected_columns = st.multiselect("Select columns", options=all_columns)

    if selected_columns:
        sub_df = df[selected_columns]
        st.write("### Sub DataFrame")
        st.write(sub_df.head())
    else:
        st.warning("Please select at least one column.")


def analyze_data(data):
    st.write("Data")
    st.write(data)

    # Group by "AÑO" and "NOM_CULTIVO" and sum "AREA_ASEG_(Has)" and "MONTO_INDN(S)" for each combination
    area_aseg_sum_by_cultivo_year = data.groupby(["AÑO", "NOM_CULTIVO"]).agg({
    "AREA_ASEG_(Has)": "sum",
    "MONTO_INDN(S)": "sum"
    }).reset_index()

# Get unique years for filtering
    years = area_aseg_sum_by_cultivo_year["AÑO"].unique()
    selected_year = st.selectbox("Seleccione un año para mostrar los totales por NOM_CULTIVO, AREA_ASEG_(Has) y MONTO_INDN(S)", years)

    st.write(f"Total de AREA_ASEG_(Has) y MONTO_INDN(S) por NOM_CULTIVO para {selected_year}-{selected_year + 1}")
    st.write(area_aseg_sum_by_cultivo_year[area_aseg_sum_by_cultivo_year["AÑO"] == selected_year][["NOM_CULTIVO", "AREA_ASEG_(Has)", "MONTO_INDN(S)"]])

    
  
    # Group by "PROVINCIA" and sum "AREA_ASEG_(Has)", "MONTO_INDN(S)", and "NUM_PROD _BENIF" for the selected year
    summary_by_provincia = data[data["AÑO"] == selected_year].groupby("PROVINCIA").agg({
        "AREA_ASEG_(Has)": "sum",
        "MONTO_INDN(S)": "sum",
        "NUM_PROD _BENIF": "sum"
    }).reset_index()

    # Display the results with handling for missing variables
    st.write(f"Totales por PROVINCIA, AREA_ASEG_(Has), MONTO_INDN(S) y NUM_PROD _BENIF para {selected_year}-{selected_year + 1}")
    st.write(summary_by_provincia.fillna("VARIABLE NO ENCONTRADA"))

    st.write("### Seleccione columnas para hacer su conjunto de datos para análisis")
    all_columns = data.columns.tolist()
    options_key = "_".join(all_columns)
    selected_columns = st.multiselect("Select columns", options=all_columns)

    if selected_columns:
        sub_df = data[selected_columns]
        st.write("### EMPECEMOS!!")
        filter_rows(sub_df)
    else:
        st.warning("Please select at least one column.")


def show_file_header(data):
    st.write("File Header")
    st.write(data.head())


def filter_rows(data):
    column_name = st.selectbox("Select a column to filter", data.columns)
    value = st.text_input("Enter the filter value")

    if value == "":
        filtered_data = data[data[column_name].isnull()]
    elif data[column_name].dtype == 'float':
        filtered_data = data[data[column_name] >= float(value)]
    else:
        filtered_data = data[data[column_name].astype(str).str.contains(value, case=False)]

    st.write("Filtered Data")
    st.write(filtered_data)

    numeric_columns = filtered_data.select_dtypes(include='number').columns
    numeric_sum = filtered_data[numeric_columns].sum()

    st.write("SUMA TOTAL DE VARIABLES PARA", column_name)
    st.write(numeric_sum)


def create_chart(chart_type, data, x_column, y_column):
    container.write(" # Data Visualization # ")
    if chart_type == "Bar":
        st.header("Bar Chart")

        color_column = st.sidebar.selectbox("Select column for color", data.columns, key="color_name")
        if color_column:
            fig = px.bar(data, x=x_column, y=y_column, color=color_column, barmode="group")
            fig.update_layout(
                title=f"{x_column} vs {y_column}",
                xaxis_title=x_column,
                yaxis_title=y_column,
                showlegend=True
            )
            st.plotly_chart(fig)
        else:
            fig = px.bar(data, x=x_column, y=y_column, barmode="group")
            fig.update_layout(
                title=f"{x_column} vs {y_column}",
                xaxis_title=x_column,
                yaxis_title=y_column,
                showlegend=True
            )
            st.plotly_chart(fig)

    elif chart_type == "Line":
        st.header("Line Chart")
        fig = px.line(data, x=x_column, y=y_column)
        fig.update_layout(
            title=f"{x_column} vs {y_column}",
            xaxis_title=x_column,
            yaxis_title=y_column
        )
        st.plotly_chart(fig)

    elif chart_type == "Scatter":
        st.header("Scatter Chart")
        size_column = st.sidebar.selectbox("Select column for size", data.columns)
        color_column = st.sidebar.selectbox("Select column for color", data.columns)
        if color_column:
            fig = px.scatter(data, x=x_column, y=y_column, color=color_column, size=size_column)
        else:
            fig = px.scatter(data, x=x_column, y=y_column)
        fig.update_layout(
            title=f"{x_column} vs {y_column}",
            xaxis_title=x_column,
            yaxis_title=y_column
        )
        st.plotly_chart(fig)

    elif chart_type == "Histogram":
        st.header("Histogram Chart")
        color_column = st.sidebar.selectbox("Select column for color", data.columns)
        fig = px.histogram(data, x=x_column, y=y_column, color=color_column)
        fig.update_layout(
            title=f"{x_column} vs {y_column}",
            xaxis_title=x_column,
            yaxis_title=y_column
        )
        st.plotly_chart(fig)

    elif chart_type == "Pie":
        st.header("Pie Chart")
        color_column = st.sidebar.selectbox("Select column for color", data.columns)
        if color_column:
            fig = px.pie(data, names=x_column, values=y_column, color=color_column)
            fig.update_layout(
                title=f"{x_column} vs {y_column}"
            )
            st.plotly_chart(fig)
        else:
            fig = px.pie(data, names=x_column, values=y_column)
            fig.update_layout(
                title=f"{x_column} vs {y_column}"
            )
            st.plotly_chart(fig)


def show_warning():
    st.sidebar.warning("AVISO: Para el análisis de datos para esta aplicación, las variables a considerar son: AÑO, PROVINCIA, DISTRITO, SECTOR_EST, NOM_CULTIVO, MONTO_INDN(S), AREA_ASEG_(Has), NUM_PROD _BENIF")


def download_file(data):
    excel_file = "cuadros.xlsx"
    download_path = os.path.join(os.path.expanduser("~"), "Downloads", excel_file)
    data.to_excel(download_path, index=False)
    return download_path


def fit_arima_model(data, column_name):
    X = data["AÑO"].values
    y = data[column_name].values

    # Ajustar modelo ARIMA
    model = sm.tsa.arima.ARIMA(y, order=(0, 1, 1), seasonal_order=(0, 1, 0, 1))
    results = model.fit()

    # Hacer predicciones para los próximos 5 años
    future_years = np.arange(data["AÑO"].max() + 1, data["AÑO"].max() + 6)
    future_predictions = results.predict(start=len(y), end=len(y) + len(future_years) - 1, typ='levels')

    # Agregar predicciones al DataFrame
    for year, prediction in zip(future_years, future_predictions):
        data = data.append({column_name: prediction, "AÑO": str(year)}, ignore_index=True)

    return data, future_predictions


def main():
    # Avatar Image
    avatar_image = Image.open("descarga2.jpg")
    st.image(avatar_image, width=700)

    st.title(":bar_chart: DIRECCIÓN DE ESTADÍSTICA AGRARIA E INFORMÁTICA (DEAI)")
    st.markdown("##")
    st.markdown('<h2 style="color: black; font-family: Georgia; font-size: 24px;">ANÁLISIS Y VISUALIZACIÓN DE DATOS</h2>', unsafe_allow_html=True)
    st.markdown('<h2 style="color: yellow; font-family: Georgia; font-size: 24px;">SEGURO AGRARIO CATASTROFICO</h2>', unsafe_allow_html=True)
    st.sidebar.image(avatar_image, width=50)
    file_option = st.sidebar.radio("Data Source", options=["Upload Local File"])
    file = None
    data = None

    if file_option == "Upload Local File":
        file = st.sidebar.file_uploader("Cargue un conjunto de datos en formato Excel", type=["xls", "xlsx"])

    options = st.sidebar.radio('Opciones', options=['Analisis de datos', 'Visualizacion de los datos', 'Series de tiempo'])

    if file is not None:
        data = load_data(file)

    if options == 'Analisis de datos':
        if data is not None:
            analyze_data(data)
        else:
            st.warning("No file or empty file")

    if options == 'Visualizacion de los datos':
        if data is not None:
            st.sidebar.title("Chart Options")

            st.write("### Select Columns")
            all_columns = data.columns.tolist()
            options_key = "_".join(all_columns)
            selected_columns = st.sidebar.multiselect("Select columns", options=all_columns)
            if selected_columns:
                sub_df = data[selected_columns]

                chart_type = st.sidebar.selectbox("Select a chart type", ["Bar", "Line", "Scatter", "Histogram", "Pie"])

                x_column = st.sidebar.selectbox("Select the X column", sub_df.columns)

                y_column = st.sidebar.selectbox("Select the Y column", sub_df.columns)

                create_chart(chart_type, sub_df, x_column, y_column)

    if options == 'Series de tiempo':
        if data is not None:
            st.sidebar.title("Time Series Options")

            time_columns = st.sidebar.multiselect("Select time column(s)", options=data.columns)
            value_columns = st.sidebar.multiselect("Select value column(s)", options=data.select_dtypes(include='number').columns)

            if time_columns and value_columns:
                time_data = data[time_columns + value_columns]
                st.write("### Time Series Data")
                st.write(time_data.head())

                selected_column = st.selectbox("Select a column for future prediction", value_columns)

                st.header("Time Series Visualization")

                fig = px.line(time_data, x="AÑO", y=selected_column, title=f"Time Series: {selected_column}")
                fig.update_layout(
                    xaxis_title="AÑO",
                    yaxis_title=selected_column
                )
                st.plotly_chart(fig)

                if st.button("Predict Future"):
                    predicted_data, future_predictions = fit_arima_model(time_data, selected_column)
                    st.write("### Predicted Data for the Next 5 Years")
                    st.write(predicted_data)

                    # Gráfico de las predicciones
                    fig_predictions = px.line(predicted_data, x="AÑO", y=selected_column, title=f"Future Predictions for {selected_column}")
                    st.plotly_chart(fig_predictions)

                    st.header("Future Predictions")
                    for i, year in enumerate(range(data["AÑO"].min() + 1, data["AÑO"].max() + 6)):
                        st.write(f"Prediction for Year {year}: {future_predictions[i]}")

    if st.button("Descargar cuadros en Excel"):
        if data is not None:
            excel_file_path = download_file(data)
            st.success(f"El archivo se ha descargado correctamente en: {excel_file_path}")


if __name__ == "__main__":
    show_warning()
    main()
