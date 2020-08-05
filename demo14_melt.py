import pandas as pd

dataFrame1 = pd.DataFrame({'Location': {0: 'Taipei', 1: 'Hsinchu', 2: "Taichung"},
                           'Java': {0: 5, 1: 10, 2: 15},
                           'Python': {0: 2, 1: 4, 2: 6},
                           'C#':{0:10, 1:20, 2:30}})
print(dataFrame1)
print("--------------------------------------")
#print(pd.melt(dataFrame1, id_vars=['Location'], value_vars=['Java']))
#print(pd.melt(dataFrame1, id_vars=['Location'], value_vars=['Java','Python']))
print(pd.melt(dataFrame1, id_vars=['Location']))