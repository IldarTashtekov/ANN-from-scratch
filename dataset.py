from random import random

def format_output(output):
    r = [0, 0]
    if output[0] > output[1]:
        r[0] = 1
    else:
        r[1] = 1

    return r

#we pass grade and calculate if aproved or not
def passed(grades):
    p1 = 0.2
    p2 = 0.4
    p3 = 0.25
    p4 = 1.0 - (p1 + p2 + p3)

    grade = grades[0] * p1 + grades[1] * p2 + grades[2] * p3 + grades[3] * p4

    if grade >= 5:
        return [1, 0]

    return [0, 1]



def generate_dataset(size):
        #generate a data set with array with grades and if the note passed or not
        data_set = [
            [
                #first we generate a array with grades
                [random()*10.0 for i in range(4)],
                []
            ] for i in range(size)
        ]

        #then we calculate if the grades are passed or not
        for sample in data_set:
            sample[1] = passed(sample[0])

        return data_set

