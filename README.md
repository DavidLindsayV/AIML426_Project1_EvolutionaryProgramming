# AIML426_Project1

In order to generate the graphs and data that are shown in the report, you should run make_figures.py
To do this:
- choose which question you want to generate tables and figures for. For example, question 5
- in the if __name__ == '__main__': statement at the bottom of make_figures, uncomment the corresponding function (and comment out any others)
For question 5, you should run q5table(). For question 4, run q4table(), etc.
- Run make_figures.py with the command  'py make_figures.py'
- If your receive an error saying that you haven't provided enough command line arguments, that's because some of the questions require command line arguments, such as what data file they should use as input. 
To provide command line arguments, simply add them at the end of your command, for example: 'py make_figures.py q1_data\knapsack-data\10_269'

A different way of running the files is running them directly. The following files can be run on their own:
- q1_knapsack.py, q2_FeatureSelect.py, q3_SymbolicRegression.py, q4_NSGA2.py, q5_CoopEA.py
These each perfrom the tasks specified in questions 1 through 5. However, they only run their algorithms once, and they do not include the same graphs and outputs as those generated by make_figures which are included in the report. 
If you want to run single runs of each algorithm, use these files. 
They may require additional command line arguments, but there are error messages that should inform you of what those command line arguments should be.


I acknowledge that the code written in this project is my own.
I acknowledge the use of the tool chatGPT for some partial assistance, but the overall program and code is my own.