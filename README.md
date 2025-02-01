# Gas Well Deliverability Calculator

This Streamlit application calculates gas well deliverability parameters using the Rawlins and Schellhardt (1936) method. It performs curve fitting on flow-after-flow test data and extrapolates to determine the Absolute Open Flow (AOF).  It also provides a manual fine-tuning option to adjust the slope and intercept of the fitted line.

## Table of Contents

- [Introduction](#introduction)
- [Theory](#theory)
- [Usage](#usage)
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This tool helps petroleum engineers and other professionals in the oil and gas industry analyze gas well deliverability. It takes flow rate and bottomhole pressure data from flow-after-flow tests, fits a line to the data in log-log space, and calculates key parameters like the performance coefficient (C) and the AOF.

## Theory

The application is based on the Rawlins and Schellhardt method, which relates flow rate to pressure drawdown in a gas well. The core equation is:

q_sc = C(P_r^2 - P_wf^2)^n

Where:

- `q_sc` is the flow rate at standard conditions.
- `C` is the performance coefficient.
- `P_r` is the reservoir pressure.
- `P_wf` is the bottomhole pressure.
- `n` is the exponent, related to the slope of the log-log plot.

The application calculates `n` and `C` by fitting a straight line to the log-log plot of `(P_r^2 - P_wf^2)` vs. `q_sc`.  It then uses these parameters to extrapolate to find the AOF, which is the theoretical flow rate when the bottomhole pressure is zero.

## Usage

1.  Run the Streamlit app (see [Installation](#installation)).
2.  Navigate to the "Gas Well Deliverability Calculator" section.
3.  Enter the flow rate (`qsc`) and bottomhole pressure (`Pwf`) data for each test point.
4.  Enter the average reservoir pressure (`Pini`).
5.  Click "Calculate".
6.  The app will display the calculated slope, intercept, performance coefficient (C), AOF, the fitted equation, the input data, and the log-log plot.
7.  To fine-tune the results, navigate to the "Deliverability Fine-tuning" section. You can manually adjust the slope (n) and intercept (b) to explore different curve fits. Click "Calculate" in this section to apply the manual adjustments and see the updated chart and extrapolated values.

## Features

-   **Data Input:**  Allows input of multiple flow rate and bottomhole pressure data points.
-   **Curve Fitting:**  Performs least-squares curve fitting to determine the relationship between flow rate and pressure.
-   **Log-Log Plot:**  Generates a log-log plot of the test data and the fitted line using Altair.
-   **AOF Calculation:**  Calculates the Absolute Open Flow (AOF) by extrapolating the fitted line.
-   **Manual Fine-tuning:** Provides an option to manually adjust the slope and intercept of the fitted line for exploration and sensitivity analysis.
-   **Clear Display:**  Presents results in a clear and organized manner, including the fitted equation, AOF, and the plot.
-   **Theory Refresher:** Provides a brief overview of the theory behind the calculations.
-   **Theoretical Example:** Shows a worked example of the calculations.

## Installation

1.  Clone the repository:

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://www.google.com/search?q=https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)  # Replace with your repo URL
cd YOUR_REPOSITORY_NAME
Create a virtual environment (recommended):
Bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required packages:
Bash
pip install -r requirements.txt
Dependencies
The application requires the following Python packages:

streamlit
pandas
numpy
scipy
altair
These are listed in requirements.txt.

File Structure
Gas_Well_Deliverability_Calculator/
├── dev_app.py          # Main Streamlit application
├── requirements.txt     # Project dependencies
├── images/             # Directory for images used in the app (image1.png, image2.png, image3.png)
└── README.md          # This file
Contributing
Contributions are welcome! Please open an issue or submit a pull request.
