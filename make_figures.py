from datetime import datetime
import sys
import matplotlib.pyplot as plt
import numpy as np

import q1_Knapsack
import q2_FeatureSelect


def make_table(data, column_labels, row_labels):
    # Create a figure and an axis
    fig, ax = plt.subplots()

    # Hide the axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=data, colLabels=column_labels, rowLabels=row_labels, loc='center')

    # Style the table
    table.auto_set_column_width(col=list(range(len(column_labels))))
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:  # Column headers (i == 0) and row labels (j == -1)
            cell.set_text_props(fontweight='bold')
        cell.set_text_props(ha='center', va='center')  # Center all text


def make_plot(data, labels, xlabel, ylabel, title):
    # Create a range of epochs
    epoch_range = range(1, len(data[0]) + 1)

    # Plot the accuracies
    plt.figure(figsize=(10, 5))
    for i in range(len(data)):
        plt.plot(epoch_range, data[i], label=labels[i])

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()


def q1table():
    if len(sys.argv) == 2:

        data = []
        plot_data = []

        seeds = [100, 200, 300, 400, 500]
        capacity, items = q1_Knapsack.process_file(sys.argv[1])
        for seed in seeds:
            best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores = q1_Knapsack.Knapsack(capacity, items, seed)
        
            plot_data.append(best_feasible_scores)

            #Calculate the best individuals weight and value sum and record them
            sum_weights = 0
            sum_value = 0
            for index, x in enumerate(best_feasible_solution):
                if x == 1:
                    sum_weights += items[index]['weight']
                    sum_value += items[index]['value']
            data.append([sum_value, sum_weights])
        
        column_labels = ["Total Value", "Total Weight"]
        row_labels = ['', '', '', '', '', 'Mean', 'Std']

        values = [row[0] for row in data]
        weights = [row[1] for row in data]
        data.append([np.mean(values), np.mean(weights)])
        data.append([np.std(values), np.std(weights)])

        make_table(data, column_labels, row_labels)
        make_plot(plot_data, ["Run 1", "Run 2", "Run 3", "Run 4", "Run 5"], "Epochs", "Total Value", "Knapsack Convergence Graph")
        plt.show()

    else:
        print('You need to input the path to the knapsack file to run')

def q2table():
    if len(sys.argv) == 2:
        classes, featurenames, data, iscontinuous, class_values, feature_values = q2_FeatureSelect.process_file(sys.argv[1])
        q2_FeatureSelect.set_global_variables(classes, featurenames, data, iscontinuous, class_values, feature_values)

        # if sys.argv[2] == 'W':
        q2_FeatureSelect.scaleFeatureValues()

        data = []
        classifier_data = []

        seeds = [100, 200, 300, 400, 500]
        for seed in seeds:
            starttime = datetime.now()
            best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores = q2_FeatureSelect.WrapperGA(seed)
            classifier_wrapper = q2_FeatureSelect.wrapper_objective(best_feasible_solution)[0]
            print("Wrapper run finished - score = " + str(best_feasible_score))
            wrappertime = round((datetime.now() - starttime).total_seconds(), 1)
            starttime = datetime.now()
            best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores = q2_FeatureSelect.FilterGA(seed)
            classifier_filter = q2_FeatureSelect.wrapper_objective(best_feasible_solution)[0]
            print("Filter run finished - score = " + str(best_feasible_score))
            filtertime = round((datetime.now() - starttime).total_seconds(), 1)
            data.append([wrappertime, filtertime])
            classifier_data.append([classifier_wrapper, classifier_filter])
        
        column_labels = ["WrapperGA Time", "FilterGA Time"]
        row_labels = ['', '', '', '', '', 'Mean', 'Std']

        wrappertime = [row[0] for row in data]
        filtertime = [row[1] for row in data]
        data.append([np.mean(wrappertime), np.mean(filtertime)])
        data.append([np.std(wrappertime), np.std(filtertime)])

        make_table(data, column_labels, row_labels)

        
        column_labels = ["WrapperGA", "FilterGA"]
        row_labels = ['', '', '', '', '', 'Mean', 'Std']

        wrapper = [row[0] for row in classifier_data]
        filter = [row[1] for row in classifier_data]
        classifier_data.append([np.mean(wrapper), np.mean(filter)])
        classifier_data.append([np.std(wrapper), np.std(filter)])

        make_table(classifier_data, column_labels, row_labels)


        plt.show()

    else:
        print('You need to input the path to the folder the GA is to run feature selection on')



if __name__ == '__main__':
    # q1table()
    q2table()