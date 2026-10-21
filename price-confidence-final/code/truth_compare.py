import json, numpy as np, subprocess
exec(open('truth_test.py').read().split('N = 400')[0])
N = 300
plans = {"flat209": [209]*26, "L129x4": [129]*4+[209]*22, "L149x4": [149]*4+[209]*22, "L169x4": [169]*4+[209]*22, "L189x4": [189]*4+[209]*22}
truth = {}
for name, q in [("good", 0.674), ("expected", 0.0), ("weak", -0.674)]:
    res = {k: np.array([run(p, q, s) for s in range(N)]) for k, p in plans.items()}
    base = res["flat209"]
    truth[name] = {k: dict(profit=float(v[:,0].mean()), diff=float((v[:,0]-base[:,0]).mean()), rev4=float(v[:,1].mean()), rating=float(v[:,2].mean())) for k, v in res.items()}
json.dump(truth, open('truth.json','w'))
print("truth done")
