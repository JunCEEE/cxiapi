import numpy as np


def distributeJob(nproc: int, ntasks: int):
    mean_tasks = ntasks // nproc
    remainder_tasks = ntasks % nproc
    process_tasks = np.full(nproc, mean_tasks)
    process_tasks[range(remainder_tasks)] += 1
    assert sum(process_tasks) == ntasks
    accu_jobs = np.add.accumulate(process_tasks)
    accu_jobs = np.insert(accu_jobs,0,0)

    return process_tasks, accu_jobs

if __name__ == "__main__":
    nproc = 12
    ntasks = 50
    jobs,accu_jobs = distributeJob(nproc,ntasks)
    print(accu_jobs)
