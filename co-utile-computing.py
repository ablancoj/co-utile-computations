import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from statistics import mean
from collections import Counter
from tqdm import tqdm


### PEER DEFINITION ###

class P:
    # w_prob is the probability of computing tasks wrong
    def __init__(self, uid, w_prob):
        self.uid = uid
        self.w_prob = w_prob
        self.managed = None
        
        self.occupied = 0
        self.times_occupied = 0
        self.local_scores = dict()
        self.global_scores = None
        
    def assign_managers(self, uids):
        self.managers = uids
        
    def assign_managed(self, uid):
        if self.managed == None:
            self.managed = [uid]
        else:
            self.managed.append(uid)
            
        if self.global_scores == None:
            self.global_scores = dict()
        
        # bootstrap reputation of unknown peer
        self.global_scores[uid] = 0.5
        
    def pass_round(self):
        self.occupied -= 1
        self.occupied = max(self.occupied, 0)

### SETTING UP THE NETWORK ###

'''
Initialize a network with n peers, where each peer
has n_SM score managers.

m_ration indicates the percentage of peers that are
malitious, and that will return wrong results with
probability m_prob. The rest of peers are considered
honest, and will return a wrong result with g_prob.

We assume that even honest peers might make mistakes
with a small probability.
'''
def init_network(n, n_SM, g_prob, m_prob, m_ratio):
    # Initialize peers
    NETWORK = {}
    malicious_peers = int(n * m_ratio)
    for i in range(n):
        if i < malicious_peers:
            NETWORK[i] = P(i, m_prob)
        else:
            NETWORK[i] = P(i, g_prob)

    # Assign managers
    peer_list = list(NETWORK.keys())
    for i in range(n):
        man = random.sample(peer_list, k=n_SM)
        # A peer cannot be its own manager
        while i in man:
            man = random.sample(peer_list, k=n_SM)
        NETWORK[i].assign_managers(man)
        for m in man:
            NETWORK[m].assign_managed(i)
    
    return NETWORK


def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

def co_utile_computation(NETWORK, SM, m, subtasks, g, random_workers, task_duration):
    
    ### PROTOCOL 1 ###

    # Needed workers pi(L)
    m_0 = m * g
    assert(m_0 <= len(NETWORK))
    
    for pi in NETWORK.keys():
        NETWORK[pi].pass_round()
    
    available_workers = list([p for p in NETWORK.keys() if NETWORK[p].occupied == 0])
    random.shuffle(available_workers)
    
    # Choose workers
    if random_workers:
        # Choose random workers
        workers = random.sample(available_workers, k=m_0)
    else:
        # Choose by score
        list_scores = [(pi, mean([NETWORK[p].global_scores[pi] for p in NETWORK[pi].managers])) for pi in available_workers]
        workers = sorted(list_scores, key=lambda x: x[1], reverse=True)[:m_0]
        workers = list([w[0] for w in workers]) 
        
    for pi in workers:
        NETWORK[pi].occupied = task_duration
        NETWORK[pi].times_occupied += 1

    # Assign tasks to workers
    TASK_ASSIGNMENT = dict()
    for tn in range(m):
        TASK_ASSIGNMENT[tn] = []
        for pn in range(g):
            TASK_ASSIGNMENT[tn].append(workers[tn*g+pn])
    
    # Complete tasks
    TASK_RESULTS = dict()
    TASK_RESULTS_MAJORITY = dict()
    for tn in range(m):
        TASK_RESULTS[tn] = []
        for pn in range(g):
            TASK_RESULTS[tn].append(subtasks[tn](NETWORK[TASK_ASSIGNMENT[tn][pn]].w_prob))
        TASK_RESULTS_MAJORITY[tn] = most_common(TASK_RESULTS[tn])

    # Reputation assignments
    for sm in SM:
        # Which peers managed by *sm* have participated in any of the tasks?
        daughters = [p for p in NETWORK[sm].managed if p in workers]
        for daughter in daughters:
            # What task was this peer assigned to?
            assigned_task = workers.index(daughter) // g
            # What other peers have the same task?
            peers_in_task = TASK_ASSIGNMENT[assigned_task]
            # What is the majority result
            majority = most_common(TASK_RESULTS[assigned_task])
            # Assign scores to the other peers participating in the same task.
            i = peers_in_task.index(daughter) 
            for j in range(g):
                if j != i:
                    other_uid = TASK_ASSIGNMENT[assigned_task][j]
                    other_result = TASK_RESULTS[assigned_task][j]
                    # Assign reputation
                    if majority == other_result:
                        NETWORK[daughter].local_scores[other_uid] = 1
                    else:
                        NETWORK[daughter].local_scores[other_uid] = 0

                    #NETWORK[daughter].local_scores[other_uid] = min(1.0, NETWORK[daughter].local_scores[other_uid])
                    #NETWORK[daughter].local_scores[other_uid] = max(0.0, NETWORK[daughter].local_scores[other_uid])
    
    ### PROTOCOL 2 ###
    
    # Score managers agree on current global reputations
    for pi in NETWORK.keys():
        ti = mean([NETWORK[man].global_scores[pi] for man in NETWORK[pi].managers])
        for man in NETWORK[pi].managers:
            NETWORK[man].global_scores[pi] = ti
    
    
    epsilon = 0.0001
    # for all Pi do
    for pi in SM:
        # for all daughters Pd of Pi (that participated in any task) do
        for pd in [p for p in NETWORK[pi].managed if p in workers]:
            # What other peers have the same task?
            peers_in_task = list(TASK_ASSIGNMENT[workers.index(pd) // g])
            peers_in_task.remove(pd)

            cjdtj = []
            for pj in peers_in_task:
                cjd = NETWORK[pj].local_scores[pd]
                tj = mean([NETWORK[man].global_scores[pj] for man in NETWORK[pj].managers])
                cjdtj.append(cjd*tj)
            td = sum(cjdtj)

            NETWORK[pi].global_scores[pd] = min(1, td) 
            #NETWORK[pi].global_scores[pd] = td
            
    # Score managers agree on current global reputations
    for pi in NETWORK.keys():
        ti = mean([NETWORK[man].global_scores[pi] for man in NETWORK[pi].managers])
        for man in NETWORK[pi].managers:
            NETWORK[man].global_scores[pi] = ti
            
    return TASK_RESULTS_MAJORITY

def run_experiment(n, n_SM, m_ratio, g_prob, m_prob, m, g, rounds, random_workers, binary_task, task_duration):
    NETWORK = init_network(n, n_SM, g_prob, m_prob, m_ratio)
    peer_list = list(NETWORK.keys())

    # List of Score managers
    SM = [p for p in peer_list if NETWORK[p].managed is not None]
   
    if binary_task:
        # Dummy code for a trivial task with probability p of returning the wrong result: True is a correct result
        task = lambda p: random.random() >= p
    else:
        # Non binary task
        task = lambda p: 0 if random.random() >= p else random.random()

    TASKS = [task]*m

    quality = []
    for _ in range(rounds):
        maj = co_utile_computation(NETWORK, SM, m, TASKS, g, random_workers, task_duration)
        if binary_task:
            q = sum([1.0 for task_result_majority in maj.values() if task_result_majority == True]) / m
        else:
            q = sum([1.0 for task_result_majority in maj.values() if task_result_majority == 0]) / m
        quality.append(q)
        
    list_scores = [mean([NETWORK[p].global_scores[pi] for p in NETWORK[pi].managers]) for pi in peer_list]
    
    return NETWORK, list_scores, mean(list_scores), mean(quality)

if __name__ == '__main__':

    pool = mp.Pool(processes=mp.cpu_count())
    
    # Probability of wrong result
    w_prob = 0.0
    assert(0.0 <= w_prob <= 1.0)

    # Number of peers    
    n = 100

    # Number of score managers
    n_SM = 3
    assert(n_SM <= n)

    # Number of subtasks
    m = 10

    # Number of rounds
    rounds = 100
	
    for g in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        for random_workers in (True, False):
            for binary_task in (True, False):
                for task_duration in (1, 2, 3):
                    if g * m * task_duration > n:
                        continue
                    test_string = f'n{n}_SM{n_SM}_m{m}_g{g}_rounds{rounds}_{"random" if random_workers else "priority"}_{"binary" if binary_task else "nonbinary"}_{task_duration}'
                    print(f'Test with parameters [{test_string}]')
                    res = []
                    qual = []
                    for p in tqdm(np.arange(0.0, 1.01, 0.01), leave=True, position=0):
                        futures = [pool.apply_async(run_experiment, (n, n_SM, p, 0.01, 1.0, m, g, rounds, random_workers, binary_task, task_duration)) for _ in range(10)]
                        res.append(mean([f.get()[2] for f in futures]))
                        qual.append(mean([f.get()[3] for f in futures]))

                    res = list(map(str, res))
                    qual = list(map(str, qual))
                    with open('coutile_computing_results_reputation.csv', 'a') as f:
                        f.write(f'({test_string}) -> {",".join(res)}\n')
                    with open('coutile_computing_results_quality.csv', 'a') as f:
                        f.write(f'({test_string}) -> {",".join(qual)}\n')
