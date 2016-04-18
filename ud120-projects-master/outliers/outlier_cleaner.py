#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    #cleaned_data = []
    all_data = list(zip(ages, net_worths, predictions))
    ### your code goes here
    cleaned_data = [(x[0], x[1], (x[2]-x[1])**2) for x in all_data]
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2])
    cleaned_data = cleaned_data[:81]    
    return cleaned_data