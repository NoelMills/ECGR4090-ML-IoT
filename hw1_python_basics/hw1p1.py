#!/usr/bin/env python
#Noel Mills
import person
import numpy as np
import matplotlib.pyplot as plt

list_of_names = ['Roger', 'Mary', 'Luisa', 'Elvis']
list_of_ages  = [23, 24, 19, 86]
list_of_heights_cm = [175, 162, 178, 182]

for name in list_of_names:
  print("The name {:} is {:} letters long".format(name, len(name)))


lengh_of_names = [len(x) for x in list_of_names]
#print (lengh_of_names)

people ={}
for i in range(len(list_of_names)):
  people[list_of_names[i]] =person.Person(list_of_names[i], list_of_ages[i], list_of_heights_cm[i])
#print(people)

age_array = np.array(list_of_ages)
height_array = np.array(list_of_heights_cm)

age_mean = np.mean(list_of_ages)
#height_mean = np.mean(list_of_heights_cm)

print(f"Average Age:{age_mean}")
#print(f"Average Height:{height_mean}")


plt.scatter(list_of_ages,list_of_heights_cm)

plt.xlabel("Age")
plt.ylabel("Height")
plt.title("Peoples Height compared to their Age")
plt.grid(True)
plt.savefig("graph.png")