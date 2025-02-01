import math
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import altair as alt
import streamlit as st


def process_data(data, Pini):
    """
    Process the provided data, perform curve fitting, and calculate necessary variables.
    
    Parameters:
    data (dict): Dictionary containing the data with keys 'Test#', 'qsc (mmscfd)', 'Pwf (psia)', and '(PR^2 - Pwf^2) (x 10^3 psia^2)'
    Pini (float): Initial reservoir pressure (psia)
    
    Returns:
    dict: A dictionary containing the slope (n), intercept (b_opt), q_AOF_extrapolated, C_extrapolated, and other calculated variables
    """
    # Create the DataFrame
    df_test = pd.DataFrame(data)

    # Exclude the first row with zero flow rate as it's not meaningful for logarithmic scale
    df_test = df_test[df_test['qsc (mmscfd)'] > 0]

    # Extract the data for curve fitting
    x_data = df_test['qsc (mmscfd)']
    y_data = df_test['(PR^2 - Pwf^2) (x 10^3 psia^2)']

    # Store the DataFrame and extracted data in the session state
    st.session_state['df_test'] = df_test
    st.session_state['x_data'] = x_data
    st.session_state['y_data'] = y_data

    # Define the model function (linear relationship)
    def linear_model(x, n, b):
        return n * x + b

    # Perform the curve fitting
    popt, pcov = curve_fit(linear_model, st.session_state['x_data'], st.session_state['y_data'])

    # Extract the optimal parameters
    n, b_opt = popt

    # Predict y values using the fitted model
    y_fit = linear_model(st.session_state['x_data'], n, b_opt)

    # Calculate residuals
    residuals = st.session_state['y_data'] - y_fit

    # Calculate ss_res and ss_tot and store them in session state
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((st.session_state['y_data'] - np.mean(st.session_state['y_data']))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Store the calculated variables in the session state
    st.session_state['n'] = n
    st.session_state['b_opt'] = b_opt
    st.session_state['y_fit'] = y_fit
    st.session_state['residuals'] = residuals
    st.session_state['ss_res'] = ss_res
    st.session_state['ss_tot'] = ss_tot
    st.session_state['r_squared'] = r_squared

    # Extrapolate to find q when y axis equals Pini^2
    Pini_squared = Pini**2  # (PR^2 - Pwf^2) when Pwf = 0
    y_value_extrapolated = Pini_squared * 1e-3  # Convert to the same units as y_data (x 10^3 psia^2)
    q_AOF_extrapolated = (y_value_extrapolated - b_opt) / n

    # Calculate the extrapolated q when y axis equals 1
    y_axis_value = 1  # y-axis value in (x 10^3 psia^2)
    C_extrapolated = (y_axis_value - b_opt) / n

    # Store the extrapolated variables in the session state
    st.session_state['q_AOF_extrapolated'] = q_AOF_extrapolated
    st.session_state['C_extrapolated'] = C_extrapolated
    st.session_state['Pini'] = Pini
    st.session_state['Pini_squared'] = Pini_squared
    st.session_state['y_axis_value'] = y_axis_value
    st.session_state['y_value_extrapolated'] = y_value_extrapolated

    return {
        'n': n,
        'b_opt': b_opt,
        'r_squared': r_squared,
        'q_AOF_extrapolated': q_AOF_extrapolated,
        'C_extrapolated': C_extrapolated,
        'x_data': x_data,
        'y_data': y_data,
        'y_fit': y_fit,
        'y_value_extrapolated': y_value_extrapolated,
    }


def plot_data_altair():
    """
    Plot the log-log curve fitting and extrapolation using Altair based on the processed data stored in session state.
    
    Returns:
    alt.Chart: The Altair chart object
    """
    # Extract data from the session state
    x_data = st.session_state['x_data']
    y_data = st.session_state['y_data']
    y_fit = st.session_state['y_fit']
    q_AOF_extrapolated = st.session_state['q_AOF_extrapolated']
    C_extrapolated = st.session_state['C_extrapolated']
    y_value_extrapolated = st.session_state['y_value_extrapolated']
    y_axis_value = st.session_state['y_axis_value']

    # Create DataFrame for the fitted line
    df_fit = pd.DataFrame({
        'qsc (mmscfd)': x_data,
        '(PR^2 - Pwf^2) (x 10^3 psia^2)': y_fit
    })

    # Create DataFrame for the extrapolated points
    df_extrapolated = pd.DataFrame({
        'qsc (mmscfd)': [C_extrapolated, q_AOF_extrapolated],
        '(PR^2 - Pwf^2) (x 10^3 psia^2)': [y_axis_value, y_value_extrapolated],
        'Label': ['C', 'AOF']
    })

    # Create the Altair chart for the original data
    chart = alt.Chart(pd.DataFrame({
        'qsc (mmscfd)': x_data,
        '(PR^2 - Pwf^2) (x 10^3 psia^2)': y_data
    })).mark_circle(size=60, color='navy').encode(
        x=alt.X('qsc (mmscfd)', scale=alt.Scale(type='log')),
        y=alt.Y('(PR^2 - Pwf^2) (x 10^3 psia^2)', scale=alt.Scale(type='log')),
        tooltip=['qsc (mmscfd)', '(PR^2 - Pwf^2) (x 10^3 psia^2)']
    ).properties(
        title='Log-Log Curve Fitting and Extrapolation',
        width=800,
        height=800
    )

    # Add the fitted line
    line = alt.Chart(df_fit).mark_line(color='red').encode(
        x='qsc (mmscfd)',
        y='(PR^2 - Pwf^2) (x 10^3 psia^2)'
    )

    # Add the extrapolated points with custom attributes
    point_C = alt.Chart(df_extrapolated[df_extrapolated['Label'] == 'C']).mark_point(
        size=100, filled=True, color='green'
    ).encode(
        x='qsc (mmscfd)',
        y='(PR^2 - Pwf^2) (x 10^3 psia^2)',
        tooltip=['qsc (mmscfd)', '(PR^2 - Pwf^2) (x 10^3 psia^2)', 'Label']
    )

    point_AOF = alt.Chart(df_extrapolated[df_extrapolated['Label'] == 'AOF']).mark_point(
        size=100, filled=True, color='orange'
    ).encode(
        x='qsc (mmscfd)',
        y='(PR^2 - Pwf^2) (x 10^3 psia^2)',
        tooltip=['qsc (mmscfd)', '(PR^2 - Pwf^2) (x 10^3 psia^2)', 'Label']
    )

    # Add text labels above the extrapolated points
    text_C = alt.Chart(df_extrapolated[df_extrapolated['Label'] == 'C']).mark_text(
        align='left', baseline='bottom', dx=5, dy=-5, color='green', fontSize=18
    ).encode(
        x='qsc (mmscfd)',
        y='(PR^2 - Pwf^2) (x 10^3 psia^2)',
        text='Label'
    )

    text_AOF = alt.Chart(df_extrapolated[df_extrapolated['Label'] == 'AOF']).mark_text(
        align='left', baseline='bottom', dx=5, dy=-5, color='orange', fontSize=18
    ).encode(
        x='qsc (mmscfd)',
        y='(PR^2 - Pwf^2) (x 10^3 psia^2)',
        text='Label'
    )

    # Add a green dashed line from y=1 to the point "C"
    green_line = alt.Chart(pd.DataFrame({
        'x': [0.1, C_extrapolated],  # Start slightly above 0 (log scale) and end at x=C_extrapolated
        'y': [y_axis_value, y_axis_value]  # Constant y=1
    })).mark_line(
        color='green', strokeDash=[5, 5]  # Dashed line
    ).encode(
        x=alt.X('x', scale=alt.Scale(type='log')),  # Ensure logarithmic scale
        y='y'
    )

    # Add an arrow-like point at the end of the green dashed line
    arrow = alt.Chart(pd.DataFrame({
        'x': [C_extrapolated],
        'y': [y_axis_value]
    })).mark_point(
        shape='triangle', size=200, angle=90, color='green'
    ).encode(
        x=alt.X('x', scale=alt.Scale(type='log')),
        y='y'
    )

    # Add a green dashed line from point "C" down to the x-axis
    green_line_down = alt.Chart(pd.DataFrame({
        'x': [C_extrapolated, C_extrapolated],
        'y': [y_axis_value, 0.1]  # Start at y_axis_value and go down to slightly above 0 (log scale)
    })).mark_line(
        color='green', strokeDash=[5, 5]  # Dashed line
    ).encode(
        x=alt.X('x', scale=alt.Scale(type='log')),  # Ensure logarithmic scale
        y=alt.Y('y', scale=alt.Scale(type='log'))  # Ensure logarithmic scale
    )

    # Add an arrow-like point at the end of the green dashed line going down to the x-axis
    arrow_down = alt.Chart(pd.DataFrame({
        'x': [C_extrapolated],
        'y': [0.1]
    })).mark_point(
        shape='triangle', size=200, angle=180, color='green'
    ).encode(
        x=alt.X('x', scale=alt.Scale(type='log')),
        y=alt.Y('y', scale=alt.Scale(type='log'))
    )

    # Add an orange dashed line from y=y_value_extrapolated to the point "AOF"
    orange_line = alt.Chart(pd.DataFrame({
        'x': [0.1, q_AOF_extrapolated],  # Start slightly above 0 (log scale) and end at x=q_AOF_extrapolated
        'y': [y_value_extrapolated, y_value_extrapolated]  # Constant y=y_value_extrapolated
    })).mark_line(
        color='orange', strokeDash=[5, 5]  # Dashed line
    ).encode(
        x=alt.X('x', scale=alt.Scale(type='log')),  # Ensure logarithmic scale
        y='y'
    )

    # Add an arrow-like point at the end of the orange dashed line
    arrow_orange = alt.Chart(pd.DataFrame({
        'x': [q_AOF_extrapolated],
        'y': [y_value_extrapolated]
    })).mark_point(
        shape='triangle', size=200, angle=90, color='orange'
    ).encode(
        x=alt.X('x', scale=alt.Scale(type='log')),
        y='y'
    )

    # Add an orange dashed line from point "AOF" down to the x-axis
    orange_line_down = alt.Chart(pd.DataFrame({
        'x': [q_AOF_extrapolated, q_AOF_extrapolated],
        'y': [y_value_extrapolated, 0.1]  # Start at y_value_extrapolated and go down to slightly above 0 (log scale)
    })).mark_line(
        color='orange', strokeDash=[5, 5]  # Dashed line
    ).encode(
        x=alt.X('x', scale=alt.Scale(type='log')),  # Ensure logarithmic scale
        y=alt.Y('y', scale=alt.Scale(type='log'))  # Ensure logarithmic scale
    )

    # Add an arrow-like point at the end of the orange dashed line going down to the x-axis
    arrow_orange_down = alt.Chart(pd.DataFrame({
        'x': [q_AOF_extrapolated],
        'y': [0.1]
    })).mark_point(
        shape='triangle', size=200, angle=180, color='orange'
    ).encode(
        x=alt.X('x', scale=alt.Scale(type='log')),
        y=alt.Y('y', scale=alt.Scale(type='log'))
    )

    # Combine the charts
    final_chart = (chart + line + point_C + point_AOF + text_C + text_AOF + green_line + arrow + green_line_down + arrow_down + orange_line + arrow_orange + orange_line_down + arrow_orange_down).interactive()

    return final_chart

# Streamlit application
st.sidebar.title("Calculator Inputs")
option = st.sidebar.radio(
    "",
    ("Theory Refresher", "Theoretical Example", "Gas Well Deliverability Calculator", "Deliverability Fine-tuning")
)


# Display content based on the selected option
if option == "Theory Refresher":

    # Set the title and subtitle with custom HTML and CSS for styling
    st.markdown("""
        <style>
        .title {
            font-family: 'Roboto', sans-serif;
            text-align: center;
            font-size: 36px;
            margin-top: 20px;
        }
        .subtitle {
            font-family: 'Roboto', sans-serif;
            text-align: center;
            font-size: 24px;
            margin-top: 10px;
        }
        </style>
        <h1 class="title">Gas Well Deliverability Calculator</h1>
        <h2 class="subtitle">After Rawlins and Schellhardt (1936).</h2>
        """, unsafe_allow_html=True)


    
    # Streamlit app content
    st.write(
        r"""
        The method is for testing gas wells by gauging flow rates against a series of back pressures. 
        
        A plot of $\boldsymbol{(P_r^2 - P_{wf}^2) = \Delta P^2}$ versus $\boldsymbol{q_{sc}}$ gives a straight line of slope $\boldsymbol{\frac{1}{n}}$ or reciprocal slope $\boldsymbol{n}$, known as the "backpressure line". The value of $ n $ may also be obtained from the angle the straight line makes with the vertical: 
        
        $\boldsymbol{n = \frac{1}{\tan \theta}}$.

        The value of performance coefficient $( C )$ is then obtained from:
        
        $\boldsymbol{C = \frac{q_{sc}}{(P_r^2 - P_{wf}^2)^n}}$
        

        The value of $\boldsymbol{C}$ can also be determined by extrapolating the straight line until the value of $\boldsymbol{(P_r^2 - P_{wf}^2)}$ is equal to **$1.0$**. The deliverability potential - Absolute Open Flow (AOF) may be obtained from the straight line (or its extrapolation) at $\boldsymbol{P_r^2}$ if $\boldsymbol{P_{wf} = 0}$ or at $\boldsymbol{(P_r^2 - P_{wf}^2)}$ when $\boldsymbol{P_{wf}}$ is the atmospheric pressure. The following equation represents the straight-line deliverability curve:
        
        $\boldsymbol{q_{sc} = C(P_r^2 - P_{wf}^2)^n}$
        
        """
    )

    # Add an image (update the path to your image)
    st.image("images/image1.png", caption="Example of flow-after-flow test data plot. Gas Well Testing Book by Amanat U. Chaundry. Page 159")



    st.write(
        r"""
        **Important to note:**
        * The value of $ n $ ranges from 0.5 to 1.0. Exponents of $ n < 0.5 $ may be caused by liquid accumulation in the wellbore.
        * Exponents apparently greater than 1.0 may be caused by fluid removal during testing. When a test is conducted using decreasing rate sequence in slow stabilizing reservoirs, an exponent greater than 1.0 may be experienced.
        * If $ n $ is outside the range of 0.5 to 1.0, the test data may be in error because of insufficient cleanup or liquid loading in the gas well.
        """
    )

    st.write(
        r"""
        **Nomencalture**:
        
        $\boldsymbol{n}$ - exponent
        
        $\boldsymbol{1/n}$ - slope
        
        $\boldsymbol{C}$ - performance coefficient
        
        $\boldsymbol{qsc}$ - rate at standart conditions, MMscfd
        
        $\boldsymbol{Pwf}$ - bottom-hole pressure, psi
        
        $\boldsymbol{Pr}$ - reservoir pressure, psi
        
        """
    )


# Display content based on the selected option
if option == "Theoretical Example":

    # Set the title and subtitle with custom HTML and CSS for styling
    st.markdown("""
        <style>
        .title {
            font-family: 'Roboto', sans-serif;
            text-align: center;
            font-size: 36px;
            margin-top: 20px;
        }
        .subtitle {
            font-family: 'Roboto', sans-serif;
            text-align: center;
            font-size: 24px;
            margin-top: 10px;
        }
        </style>
        <h1 class="title">Gas Well Deliverability Calculator</h1>
        <h2 class="subtitle">Stabilized Flow Test Analysis</h2>
        """, unsafe_allow_html=True)
    

    st.write(
        r"""
            A flow-after-flow test was performed on a gas well located in a low-pressure
            reservoir. Using the following test data, determine the values of n and C for
            the deliverability equation, AOF.
        """
    )

    # Add an image (update the path to your image)
    st.image("images/image2.png", caption="Flow-after-Flow Test Data. Gas Well Testing Book by Amanat U. Chaundry. Page 160")
    
        # Add an image (update the path to your image)
    st.image("images/image3.png", caption="Flow-after-Flow Test Data. Gas Well Testing Book by Amanat U. Chaundry. Page 161")

if option == "Gas Well Deliverability Calculator":
    st.write("**Gas Well Test Inputs**")

    # Initialize session state for data if not already set
    if 'data' not in st.session_state:
        st.session_state['data'] = {
            "qsc (mmscfd)": [0, 0, 0, 0],
            "Pwf (psia)": [0, 0, 0, 0]
        }

    data = {
        "qsc (mmscfd)": [],
        "Pwf (psia)": []
    }

    # Collect input data from the user
    for i in range(len(st.session_state['data']["qsc (mmscfd)"])):
        col1, col2 = st.columns(2)
        with col1:
            qsc_label = f"qsc (mmscfd) - Row {i+1}"
            qsc = st.number_input(qsc_label, value=float(st.session_state['data']["qsc (mmscfd)"][i]), format="%.3f")
        with col2:
            pwf_label = f"Pwf (psia) - Row {i+1}"
            pwf = st.number_input(pwf_label, value=float(st.session_state['data']["Pwf (psia)"][i]), format="%.1f")
        data["qsc (mmscfd)"].append(qsc)
        data["Pwf (psia)"].append(pwf)

    # Update session state with new data
    st.session_state['data'] = data

    df = pd.DataFrame(st.session_state['data'])

    # Display the updated DataFrame
    st.write("**Updated Test Inputs:**")
    st.table(df)

    # Add the initial reservoir pressure input
    if 'Pini' not in st.session_state:
        st.session_state['Pini'] = 0.0

    Pini = st.number_input("Average Reservoir Pressure (Pini - psia), gas surface would be zero", value=st.session_state['Pini'], step=0.1)
    st.session_state['Pini'] = Pini

    # Add a "Calculate" button
    if st.button("Calculate"):
        # Prepare the data dictionary for processing
        data_dict = {
            'Test#': list(range(1, len(df) + 1)),
            'qsc (mmscfd)': df["qsc (mmscfd)"],
            'Pwf (psia)': df["Pwf (psia)"],
            '(PR^2 - Pwf^2) (x 10^3 psia^2)': (Pini**2 - df["Pwf (psia)"]**2) * 1e-3
        }

        st.session_state.df = df

        # Process the data
        processed_data = process_data(data_dict, Pini)

        # Plot the data
        final_chart = plot_data_altair()

        # Extract data from the session state
        q_AOF_extrapolated = st.session_state['q_AOF_extrapolated']
        C_extrapolated = st.session_state['C_extrapolated']
        n = st.session_state['n']
        b_opt = st.session_state['b_opt']
        


        # Display the chart and data
        st.write(f"Slope:  {st.session_state['n']:.3f}")
        st.write(f"Intercept:  {st.session_state['b_opt']:.3f}")
        st.write(f"Constant C:  {st.session_state['C_extrapolated']:.3f}")
        st.write(f"Fitted equation:  y = {n:.3f}x + {b_opt:.3f}")
        st.write(f"AOF:  {st.session_state['q_AOF_extrapolated']:.2f} mmscfd")
        
        st.write(df)
        st.altair_chart(final_chart)

        
if option == "Deliverability Fine-tuning":
    # Check if session state variables are set before accessing them
    if 'n' not in st.session_state or 'b_opt' not in st.session_state or 'df' not in st.session_state:
        st.write("Please run the calculation first to set the slope (n) and intercept (b_opt).")
    else:
        n = st.session_state['n']
        b_opt = st.session_state['b_opt']
        df = st.session_state['df']

        # Initializing variables if not already set
        if 'n_adjusted' not in st.session_state:
            st.session_state['n_adjusted'] = 0.00
        if 'b_adjusted' not in st.session_state:
            st.session_state['b_adjusted'] = 0.00

        # Set the title and subtitle with custom HTML and CSS for styling
        st.markdown("""
            <style>
            .title {
                font-family: 'Roboto', sans-serif;
                text-align: center;
                font-size: 36px;
                margin-top: 20px;
            }
            .subtitle {
                font-family: 'Roboto', sans-serif;
                text-align: center;
                font-size: 24px;
                margin-top: 10px;
            }
            </style>
            
            <h2 class="subtitle">More often than not, it is useful to explore fitting the curve manually.</h2>
            """, unsafe_allow_html=True)

        st.write("Below there is an option to adjust slope (m) and intercept (b) in the linear fit equation:")
        st.latex(r"y = nx + b \quad \text{where} \quad n = \text{slope}, \quad b = \text{intercept}")

        # Create a single column for layout
        col1, col2 = st.columns([1, 3])

        with col2:
            # Adding input boxes in a single column
            st.subheader("Fitted values:")
            st.write(f"Slope (n): {n:.3f}")
            st.write(f"Intercept (b): {b_opt:.3f}")
            
            st.subheader("Manual Adjustments")
            st.session_state['n_adjusted'] = st.number_input("Adjusted slope n", min_value=0.00, max_value=2.00, step=0.01, value=st.session_state['n_adjusted'])
            st.session_state['b_adjusted'] = st.number_input("Adjusted intercept b", min_value=-100.00, max_value=100.00, step=0.01, value=st.session_state['b_adjusted'])

        Pini = st.session_state['Pini']
        
        
        df_test = st.session_state['df_test']
        x_data = st.session_state['x_data']
        y_data = st.session_state['y_data']        
        
        
        #st.write(df_test)     
        #st.write(x_data)     
        #st.write(y_data)  
        
        #st.write(type(x_data).__name__)
        #st.write(type(y_data).__name__)
        
        # Combine the series into a DataFrame
        df_charting = pd.DataFrame({
            'qsc (mmscfd)': x_data,
            '(PR^2 - Pwf^2) (x 10^3 psia^2)': y_data
        })

        #st.write(df_charting)  
        

        # Add a "Calculate" button
        if st.button("Calculate"):
            def manual_process_and_plot_data_altair(df, Pini, n_adjusted, b_adjusted):
                # Filter for positive qsc values
                df_test = df[df['qsc (mmscfd)'] > 0]

                # Extract x and y data
                x_data_manual = x_data
                y_data_manual = y_data

                # Define the manual linear model
                def manual_linear_model(x, n_adjusted, b_adjusted):
                    return n_adjusted * x + b_adjusted

                # Perform the curve fitting
                popt1, pcov1 = curve_fit(manual_linear_model, x_data_manual, y_data_manual)
                n_opt1, b_opt1 = popt1  # Store the optimized values

                # Calculate extrapolated values
                Pini_squared = Pini**2
                y_value_extrapolated = Pini_squared * 1e-3
                q_AOF_extrapolated_manual = (y_value_extrapolated - b_opt1) / n_opt1 
                y_axis_value = 1
                C_extrapolated_manual = (y_axis_value - b_opt1) / n_opt1

                # Store the calculated variables in the session state
                st.session_state['n_adjusted'] = n_adjusted
                st.session_state['b_adjusted'] = b_adjusted
                st.session_state['q_AOF_extrapolated_manual'] = q_AOF_extrapolated_manual
                st.session_state['C_extrapolated_manual'] = C_extrapolated_manual

                # Display the fitted equation and extrapolated values
                st.write(f"Fitted equation: y = {n_adjusted:.3f}x + {b_adjusted:.3f}")
                st.write(f"Extrapolated AOF: {q_AOF_extrapolated_manual:.3f}")
                st.write(f"Extrapolated constant C: {C_extrapolated_manual:.3f}")

                # Create the Altair chart for the original data
                chart = alt.Chart(df_charting).mark_circle(size=60, color='navy').encode(
                    x=alt.X('qsc (mmscfd)', scale=alt.Scale(type='log')),
                    y=alt.Y('(PR^2 - Pwf^2) (x 10^3 psia^2)', scale=alt.Scale(type='log')),
                    tooltip=['qsc (mmscfd)', '(PR^2 - Pwf^2) (x 10^3 psia^2)']
                ).properties(
                    title='Log-Log Curve Fitting and Extrapolation',
                    width=800,
                    height=800
                )

                # Add the fitted line to the chart
                line = alt.Chart(df_charting).mark_line(color='red').encode(
                    x=alt.X('qsc (mmscfd)', scale=alt.Scale(type='log')),
                    y=alt.Y('linear_model:Q', scale=alt.Scale(type='log'))
                ).transform_calculate(
                    linear_model=f'datum["qsc (mmscfd)"] * {n_adjusted} + {b_adjusted}'
                )

                # Combine the charts
                final_chart = (chart + line).interactive()

                # Display the chart in Streamlit
                st.altair_chart(final_chart, use_container_width=True)

            # Extract the DataFrame from the session state
            
            manual_process_and_plot_data_altair(df, Pini, st.session_state['n_adjusted'], st.session_state['b_adjusted'])