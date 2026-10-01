import argparse
import json
import time
from pathlib import Path
import numpy as np
from common import BASE,finish,save_json,table,plt
from model import Problem,solve


def main():
    p=argparse.ArgumentParser(description='Вариант 16: эволюционный отбор тестов')
    p.add_argument('--config',type=Path,default=BASE/'config.json');p.add_argument('--runs',type=int);p.add_argument('--seed',type=int);p.add_argument('--output',type=Path,default=BASE/'results')
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'))
    if a.runs is not None:c['runs']=a.runs
    if a.seed is not None:c['first_seed']=a.seed
    if c['runs']<1 or c['population']<2 or c['budget']<c['population'] or c['first_seed']<0 or not all(0<=c[k]<=1 for k in ['mutation_probability','crossover_probability','local_probability']):p.error('Некорректная конфигурация')
    data=json.loads((BASE/'data/tests.json').read_text(encoding='utf-8'));problem=Problem(data)
    rows=[];traces={};solutions=[]
    for method,memetic in [('GA',False),('Memetic-GA',True)]:
        traces[method]=[]
        for i in range(c['runs']):
            seed=c['first_seed']+i;start=time.perf_counter();score,x,trace,calls,local=solve(c,problem,seed,memetic)
            coverage,duration=problem.metrics(x)
            rows.append(dict(method=method,seed=seed,score=score,coverage=coverage,coverage_fraction=coverage/int(problem.weights.sum()),duration=duration,test_count=int(x.sum()),seconds=time.perf_counter()-start,evaluations=calls,local_evaluations=local,feasible=int(duration<=problem.limit)))
            traces[method].append(trace);solutions.append(dict(method=method,seed=seed,mask=x.astype(int).tolist(),selected=[name for name,chosen in zip(data['tests'],x) if chosen],score=score,coverage=coverage,duration=duration))
        print(method,'completed',flush=True)
    finish(a.output,c,rows,traces,'Best weighted coverage minus time penalty (maximize)')
    save_json(a.output/'solutions.json',solutions)
    greedy=problem.greedy();coverage,duration=problem.metrics(greedy)
    save_json(a.output/'greedy.json',dict(selected=[t for t,s in zip(data['tests'],greedy) if s],score=problem.objective(greedy),coverage=coverage,duration=duration,coverage_upper_bound=int(problem.weights.sum())))
    best=max(solutions,key=lambda x:x['score']);chosen=np.array(best['mask'],dtype=bool)
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained');ax.imshow(problem.matrix[chosen],cmap='Blues',aspect='auto',interpolation='nearest');ax.set_yticks(range(chosen.sum()),np.array(data['tests'])[chosen]);ax.set_xticks(range(36),data['requirements'],rotation=90,fontsize=7);ax.set_title(f"Selected tests: coverage={best['coverage']}, duration={best['duration']}/70");ax.set_xlabel('Requirements');fig.savefig(a.output/'coverage.png',dpi=160);plt.close(fig)
    differences=np.array([r['score'] for r in rows if r['method']=='Memetic-GA'])-np.array([r['score'] for r in rows if r['method']=='GA'])
    rng=np.random.default_rng(50717);boot=rng.choice(differences,(10000,len(differences)),replace=True).mean(axis=1)
    save_json(a.output/'paired_comparison.json',dict(mean_difference=float(differences.mean()),median_difference=float(np.median(differences)),bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),wins=int((differences>1e-10).sum()),ties=int((abs(differences)<=1e-10).sum()),losses=int((differences< -1e-10).sum()),bootstrap_seed=50717,bootstrap_samples=10000))

if __name__=='__main__':main()
