from joblib import dump

x = []

for i in range(100):
    
    x.append(i/2)

dump(x, 'x.joblib')