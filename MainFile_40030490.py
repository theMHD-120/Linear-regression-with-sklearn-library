import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""
|| in the name of ALLAH ||
the Final Project, Numerical Methods
<<< Implementing a linear regression >>>
seyed mahdi mahdavi mortazavi
stdNo: 40030490
"""

print("\n||| --------------------------- In the name of ALLAH ---------------------------- |||")
print("Implementing a linear regression algorithm with python;")
print("In this project, we predicate a person's Age, with his/her body measurements;")
print("Note: You can download data file from this address:")
print("https://www.kaggle.com/datasets/saurabhshahane/body-measurements-dataset")
input("Enter to continue ...")

y_list: list  # a list for dependent variable
x_list: list  # a list for independent variables
body_measures = pd.read_csv('Body Measurements _ original_CSV.csv')


print("\n||| --------------------- Step 1: General file information ---------------------- |||")
print('1) The first 5 rows ----------------------------')
print(body_measures.head())
print('\n2) File information --------------------------')
print(body_measures.info())
print('\n3) Statistical information -------------------')
print(body_measures.describe())


print("\n||| ----------------------- Step 2: Make regression line ------------------------ |||")
x_list = body_measures[
    ["Gender", "HeadCircumference", "ShoulderWidth", "ChestWidth", "Belly", "Waist", "Hips",
     "ArmLength", "ShoulderToWaist", "WaistToKnee", "LegLength", "TotalHeight"]]
y_list = body_measures["Age"]
reg_res = LinearRegression()
reg_res.fit(x_list, y_list)

i = 0  # index of coefficients
print("The coefficient of variables are shown in bottom...")
for header in body_measures.columns:
    if header != "Age":
        print(f"{header}: {reg_res.coef_[i]}")
        i += 1
print("-----------------------------------")
print("Intercept of regression line is: ", reg_res.intercept_)

# Predict Ages and calculate error -------------------------------------------------
predicted_y = reg_res.predict(x_list)  # predicated y list (y is list Ages)
rmse = np.sqrt(np.mean(np.square(predicted_y - y_list)))  # root mean square error
print("Average prediction error: ", rmse, " ~= ", round(rmse))


print("\n||| -------------------------- Step 3: Solve an example ------------------------- |||")
# You can use this numbers for example: 1, 20, 25, 24, 25, 36, 17, 23, 12, 9, 19, 48
x_for_predict = [[]]
for header in x_list:
    x_for_predict[0].append(int(input(f"Enter a number for input var <{header}>: ")))
predicted_example = reg_res.predict(x_for_predict)[0]
print("An example to prediction: ", abs(predicted_example), " ~= ", round(abs(predicted_example)), "years old")


print("\n||| -------------------------- Step 4: Draw the plots --------------------------- |||")
for header in x_list:
    plt.grid()
    # New regression for per input
    reg_res = LinearRegression()
    reg_input = x_list[header].values.reshape(-1, 1)
    reg_res.fit(reg_input, y_list)
    y_reg_line = reg_res.coef_ * x_list[header] + reg_res.intercept_
    # The plot inputs
    plt.scatter(x_list[header], y_list, color="purple")
    plt.scatter(x_list[header], predicted_y, color="orange")
    plt.plot(x_list[header], y_reg_line, color="red")
    # The plot styles
    plt.xlabel(header, fontsize=15)
    plt.ylabel("Age", fontsize=15)
    plt.legend(["Original", "Predicted", "Regression line"])
    plt.show()


print("\n||| ---------------------------------- The End ---------------------------------- |||")
print("About me:")
print("Seyed mahdi mahdavi mortazavi")
print("Student of CSE, Shiraz university")
print("My github link: https://github.com/theMHD-120")
print("Good luck! :)")