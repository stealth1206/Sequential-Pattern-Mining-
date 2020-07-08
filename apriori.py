import pandas as pd    # Importing pandas
from mlxtend.preprocessing import TransactionEncoder  # Importing transaction encoder
from mlxtend.frequent_patterns import apriori   # Importing apriori algorithm
from mlxtend.frequent_patterns import association_rules   # Importing association rules for associtaion rules mining
import matplotlib.pylab as plt
import time

def check_dict(dict,key):    # Function for checking if key is already present in Dictionary
	if key in dict.keys(): 
		return True 
	else: 
		return False

start = time.time()
dataset = []   # 2-D array for storing the sequences
with open('out.txt', 'r') as fobj:    # Importing values from txt file containing dataset
    for line in fobj:
        numbers = [int(num) for num in line.split()]   # Single row of the 2-D array
        dataset.append(numbers)

t = TransactionEncoder()     
t_ary = t.fit(dataset).transform(dataset)  # Convrerting to table of true/false
df = pd.DataFrame(t_ary, columns=t.columns_)    # Converting t_ary table to suitable form for giving input to apriori
frequent_set = apriori(df, min_support=0.015,use_colnames=True)   # Applying apriori algorithm

frequent_set['length'] = frequent_set['itemsets'].apply(lambda x: len(x))
print(frequent_set)

rules = association_rules(frequent_set,metric='lift')     # Applying association rules 
#print(rules)

end = time.time()
#print(end-start)

# For generating length v/s Count plot

# d={}  # Intializing a dictionary
# for i in range(frequent_set.shape[0]):
# 	if check_dict(d,frequent_set['length'][i]):
# 		d[frequent_set['length'][i]] = d[frequent_set['length'][i]]+1
# 	else :
# 		d[frequent_set['length'][i]] = 1;


# lists = sorted(d.items()) # sorted by key, return a list of tuples

# x, y = zip(*lists) # unpack a list of pairs into two tuples

# plt.style.use('ggplot') 
# plt.title('Count v/s Length Plot')
# plt.xlabel("Length")
# plt.ylabel("Count")
# plt.plot(x, y)   # Plotting the plot
# plt.show()
