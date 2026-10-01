"""Отбор тестов: ГА и меметический ГА с единым счётчиком оценок."""
import numpy as np
from common import tournament


def generate(seed=2650717):
    rng=np.random.default_rng(seed);n=50;m=36
    matrix=rng.random((n,m))<.10
    for j in range(m):matrix[rng.integers(n),j]=True
    return dict(seed=seed,tests=[f'T{i+1:02}' for i in range(n)],requirements=[f'R{j+1:02}' for j in range(m)],coverage=matrix.astype(int).tolist(),duration=rng.integers(3,16,n).tolist(),weights=rng.integers(1,6,m).tolist(),time_limit=70)


class Problem:
    def __init__(self,data):
        self.matrix=np.array(data['coverage'],dtype=bool)
        self.duration=np.array(data['duration']);self.weights=np.array(data['weights']);self.limit=data['time_limit']

    def metrics(self,x):
        covered=self.matrix[x].any(axis=0) if np.any(x) else np.zeros(self.matrix.shape[1],dtype=bool)
        return int(self.weights@covered),int(self.duration@x)

    def objective(self,x):
        coverage,duration=self.metrics(x)
        if duration>self.limit:raise ValueError('Недопустимый тестовый набор')
        # Разница покрытия >=1 всегда важнее всей разницы длительности <=0.1.
        return coverage-.1*duration/self.limit

    def repair(self,x):
        x=np.asarray(x,dtype=bool).copy()
        # Ремонт использует статический приоритет, не скрытый поиск по фитнесу.
        static=(self.matrix@self.weights)/self.duration
        for i in np.argsort(static,kind='stable'):
            if self.duration@x<=self.limit:break
            x[i]=False
        return x

    def greedy(self):
        x=np.zeros(len(self.duration),dtype=bool);covered=np.zeros(len(self.weights),dtype=bool)
        while True:
            candidates=np.flatnonzero(~x & (self.duration+self.duration@x<=self.limit))
            if not len(candidates):break
            gain=(self.matrix[candidates]&~covered)@self.weights
            if gain.max()==0:break
            i=candidates[np.argmax(gain/self.duration[candidates])];x[i]=True;covered|=self.matrix[i]
        return x


def solve(c,problem,seed,memetic=False):
    rng=np.random.default_rng(seed);n=c['population'];d=len(problem.duration);budget=c['budget']
    best=None;best_value=-np.inf;calls=0;trace=[]
    checkpoints=set(range(n,budget+1,n));checkpoints.add(budget)
    def evaluate(x):
        nonlocal calls,best,best_value
        if calls>=budget:raise RuntimeError('Budget exceeded')
        value=problem.objective(x);calls+=1
        if value>best_value:best_value=value;best=x.copy()
        if calls in checkpoints:trace.append(float(best_value))
        return value
    pop=np.array([problem.repair(x) for x in (rng.random((n,d))<.18)])
    values=np.array([evaluate(x) for x in pop]);local_calls=0
    while calls<budget:
        children=[];scores=[]
        for _ in range(min(n,budget-calls)):
            a,b=tournament(rng,-values,2);child=pop[a].copy()
            if rng.random()<c['crossover_probability']:
                mask=rng.random(d)<.5;child[mask]=pop[b][mask]
            child^=rng.random(d)<c['mutation_probability'];child=problem.repair(child)
            value=evaluate(child)
            if memetic and rng.random()<c['local_probability']:
                # До 8 случайных битовых соседей; каждое улучшение принимается.
                for bit in rng.permutation(d)[:8]:
                    if calls>=budget:break
                    neighbor=child.copy();neighbor[bit]=~neighbor[bit];neighbor=problem.repair(neighbor)
                    candidate=evaluate(neighbor);local_calls+=1
                    if candidate>value:child,value=neighbor,candidate
            children.append(child);scores.append(value)
            if calls>=budget:break
        both=np.concatenate([pop,np.array(children)]);all_values=np.r_[values,scores]
        order=np.argsort(-all_values,kind='stable')[:n];pop,values=both[order],all_values[order]
    return best_value,best,trace,calls,local_calls
