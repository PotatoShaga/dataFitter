import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline #actually maybe lines not curving isnt a problem

#Questions: is state session/saving needed for fully marks? how many equations are needed? is this graph enough info?
#and just local host is fine?

#st.radio for picking fit
"Input values manually or attach csv"
cols = st.columns(3,vertical_alignment="top") #this is a action. makes 3 columnns (and I fill them) after the above sentence

with cols[0]: #1st column, empty_df
    "Insert values manually here:"
    if "datapoint_number" not in st.session_state: #this if block sets the default, and also keeps previous data in the st.session_state, this is in dict format
        st.session_state["datapoint_number"] = 5 

    empty_df_size = st.text_input(label="Number of datapoints:", value=st.session_state["datapoint_number"])
    if (not empty_df_size.isdigit()) or (int(empty_df_size) < 1):
        'Positive integer values only. 10 data points created.'
        empty_df_size = 9 
    empty_df_size = int(empty_df_size) + 1
    init_dict = {x:0.0 for x in range(1,empty_df_size)}
    empty_df = pd.DataFrame(init_dict, index=["x","y"]) #initialising empty df
    empty_df = empty_df.transpose()
    empty_df["x"] = empty_df.index #sets x column to index so that polyfit doesnt die immediately with empty df
    df = st.data_editor(empty_df)

using_csv_data = False

with cols[1]: #2nd column, file dropbox
    "File Dropbox:"
    with st.form("csv_form"):
        csv = st.file_uploader(label=".csv", type=".csv")

        submitted = st.form_submit_button("Submit")
        if submitted:
            if csv == None:
                "NO csv attached"
            else:
                df1 = pd.read_csv("test.csv", header=None)
                df1.columns = ["x", "y"]
                df = df1 #once submit button is pressed the csv is passed to df
                using_csv_data = True

with cols[2]: #3rd column, df to be fit
    "DATA TO BE PLOTTED: "
    if using_csv_data == True:
        "Using attached csv values:"
    df


#LOWER PORTION: radio fit selection and visual graph
#Excel has trendlines: exponential, linear, logarithmic, polynomial (deg N), power
fit_selection = st.radio(label="Choose your fit:", options=["Linear", "Polynomial", "Exponential", "Logarithmic"])

if fit_selection == "Linear":
    trendline = np.polyfit(df["x"], df["y"], deg=1) #generates coefficients for linear function, m and b
    trendline_y_values = np.polyval(trendline, df["x"]) #calculates y values with given coefficients
    plt.scatter(df["x"], df["y"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(df["x"],trendline_y_values)
    st.pyplot(plt)

    equation = f"{round(trendline[0], 5)}x + {round(trendline[1], 5)}"
    equation

elif fit_selection == "Polynomial":
    degree = st.slider("Degree:", min_value = 2, max_value = 10, value=2)

    trendline = np.polyfit(df["x"], df["y"], deg=int(degree))
    trendline_y_values = np.polyval(trendline, df["x"])
    plt.scatter(df["x"], df["y"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(df["x"],trendline_y_values)
    st.pyplot(plt)
    
    trendline = [round(trendline[i], 5) for i,x in enumerate(trendline)]
    equation = "y = "
    for x in range(degree+1, 0, -1):
        if x == 1: #just makes the last string added not have a trailing +
            equation += f"({list(reversed(trendline))[x-1]})x^{x-1}"
        else:
            equation += f"({list(reversed(trendline))[x-1]})x^{x-1} + "
    equation

elif fit_selection == "Exponential":
    def exponential(x, A, b, c):
        y = A * np.exp(b*x) + c #np.exp is a function that takes b,x as input
        return y
    coeffs = curve_fit(exponential, xdata=df["x"], ydata=df["y"], p0=(1,1,1))

    trendline_y_values = coeffs[0][0] * np.exp(coeffs[0][1] * df["x"]) + coeffs[0][2]
    plt.scatter(df["x"], df["y"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(df["x"],trendline_y_values)

    st.pyplot(plt)

    equation = f"y = {round(coeffs[0][0], 5)} * e^({round(coeffs[0][1], 5)}x) + ({round(coeffs[0][2], 5)})"
    equation

elif fit_selection == "Logarithmic":
    def logarithmic(x, A, b):
        y = A * np.log(x) + b #np.exp is a function that takes b,x as input
        return y
    coeffs = curve_fit(logarithmic, xdata=df["x"], ydata=df["y"])

    trendline_y_values = coeffs[0][0] * np.log(df["x"]) + coeffs[0][1]
    plt.scatter(df["x"], df["y"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(df["x"],trendline_y_values)
    st.pyplot(plt)

    equation = f"y = {round(coeffs[0][0], 5)} * log(x) + ({round(coeffs[0][1], 5)})"
    equation

