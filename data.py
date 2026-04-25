import pandas as pd
import numpy as np

np.random.seed(42)

def generate_titanic(n, include_survived=True, start_id=1):
    pclass = np.random.choice([1,2,3], n, p=[0.24, 0.21, 0.55])
    sex = np.random.choice(['male','female'], n, p=[0.65, 0.35])
    
    ages = []
    for i in range(n):
        if np.random.rand() < 0.20:
            ages.append(np.nan)
        else:
            a = float(np.clip(np.random.gamma(3, 10) + 2, 0.5, 80))
            ages.append(round(a, 1))
    age_raw = np.array(ages, dtype=float)
    
    sibsp = np.random.choice([0,1,2,3,4,5,8], n, p=[0.682,0.230,0.050,0.020,0.010,0.005,0.003])
    parch = np.random.choice([0,1,2,3,4,5,6], n, p=[0.760,0.130,0.080,0.010,0.010,0.005,0.005])
    
    fare_base = np.where(pclass==1, np.random.lognormal(4.5,0.8,n),
                np.where(pclass==2, np.random.lognormal(3.0,0.5,n),
                                    np.random.lognormal(2.2,0.7,n)))
    fare_base = np.clip(fare_base, 0, 512)
    outlier_idx = np.random.choice(n, int(n*0.03), replace=False)
    fare_base[outlier_idx] = np.random.uniform(300, 512, len(outlier_idx))
    
    cabin = []
    for i in range(n):
        if pclass[i] == 1 and np.random.rand() < 0.6:
            cabin.append(f"{np.random.choice(['A','B','C','D'])}{np.random.randint(1,100)}")
        elif pclass[i] == 2 and np.random.rand() < 0.1:
            cabin.append(f"{np.random.choice(['D','E','F'])}{np.random.randint(1,100)}")
        else:
            cabin.append(None)
    
    embarked_vals = ['S','C','Q']
    emb_raw = np.random.choice([0,1,2,3], n, p=[0.720,0.190,0.086,0.004])
    embarked = [embarked_vals[e] if e < 3 else None for e in emb_raw]
    
    first_m = ['James','John','William','Charles','George','Thomas','Arthur','Henry','Frederick','Edward']
    last_n = ['Smith','Johnson','Williams','Jones','Brown','Davis','Miller','Wilson','Moore','Taylor',
              'Anderson','Thomas','Jackson','White','Harris','Martin','Thompson','Garcia','Martinez','Robinson']
    titles_m = ['Mr.','Mr.','Mr.','Mr.','Mr.','Mr.','Rev.','Dr.','Col.','Capt.']
    first_f = ['Mary','Anna','Margaret','Elizabeth','Helen','Alice','Ethel','Frances','Rose','Florence']
    titles_f = ['Miss','Miss','Miss','Mrs.','Mrs.','Mrs.','Mrs.','Lady.','Countess.','Dr.']
    
    names = []
    for i in range(n):
        ln = np.random.choice(last_n)
        if sex[i] == 'male':
            fn = np.random.choice(first_m)
            t = np.random.choice(titles_m)
        else:
            fn = np.random.choice(first_f)
            t = np.random.choice(titles_f)
        names.append(f"{ln}, {t} {fn}")
    
    ticket = [str(np.random.randint(10000,999999)) for _ in range(n)]
    
    data = {
        'PassengerId': list(range(start_id, start_id+n)),
        'Pclass': pclass.tolist(),
        'Name': names,
        'Sex': sex.tolist(),
        'Age': age_raw.tolist(),
        'SibSp': sibsp.tolist(),
        'Parch': parch.tolist(),
        'Ticket': ticket,
        'Fare': fare_base.round(4).tolist(),
        'Cabin': cabin,
        'Embarked': embarked
    }
    
    if include_survived:
        surv_prob = np.zeros(n)
        for i in range(n):
            p = 0.3
            if sex[i] == 'female': p += 0.4
            if pclass[i] == 1: p += 0.2
            elif pclass[i] == 2: p += 0.1
            if not np.isnan(age_raw[i]) and age_raw[i] < 16: p += 0.2
            surv_prob[i] = min(p, 0.95)
        survived = (np.random.rand(n) < surv_prob).astype(int).tolist()
        data['Survived'] = survived
        cols = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    else:
        cols = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    
    return pd.DataFrame(data)[cols]

train = generate_titanic(891, include_survived=True)
test = generate_titanic(418, include_survived=False, start_id=892)

train.to_csv('/home/claude/titanic_assignment/data/train.csv', index=False)
test.to_csv('/home/claude/titanic_assignment/data/test.csv', index=False)
print("Train:", train.shape, "| Survival rate:", round(train['Survived'].mean(), 3))
print("Test:", test.shape)
print("\nMissing values (train):")
print(train.isnull().sum())
