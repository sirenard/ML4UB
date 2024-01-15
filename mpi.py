from collections import namedtuple
from enum import Enum

from mpi4py import MPI

# Need to do it from scratch, advanced libs don't work on the targeted cluster

class MpiPool:
    def __init__(self):
        mpi_comm = MPI.COMM_WORLD
        self.rank = mpi_comm.Get_rank()
        self.world_size = mpi_comm.Get_size()

        self.workers = None

        if self.rank == 0:
            self.workers = [Worker(rank) for rank in range(1, self.world_size)]


    def __enter__(self):
        if self.rank == 0:
            return self
        else:
            self.worker_loop()
            return None

    def __exit__(self, *_):
        if self.rank:
            return None

        mpi_comm = MPI.COMM_WORLD
        for worker in self.workers:
            mpi_comm.send((None,None), worker.rank, tag=0)

    def map(self, f, args):
        if self.world_size == 1:
            return [f(*arg) for arg in args]

        res = [None for _ in args]
        indexes = [0 for _ in self.workers]
        mpi_comm = MPI.COMM_WORLD
        status = MPI.Status()

        index = 0
        receive_counter = 0
        while True:
            worker = self.get_ready_worker()

            if receive_counter == len(args):
                break

            if worker is not None and index < len(args): # send job to worker
                rank = worker.rank
                worker.set_working()
                mpi_comm.send((f,args[index]), rank, tag=1)
                indexes[rank-1] = index
                index += 1

            else: # no worker available, wait for jobs to finish
                val = mpi_comm.recv(status=status)
                rank = status.Get_source()

                worker = self.workers[rank-1]
                worker.set_ready()

                i = indexes[rank-1]
                res[i] = val
                receive_counter += 1

        assert "error" not in res

        return res

    def worker_loop(self):
        mpi_comm = MPI.COMM_WORLD
        status = MPI.Status()
        rank = mpi_comm.Get_rank()


        done = False
        while not done:
            mpi_comm.Probe()
            (f, args) = mpi_comm.recv(status=status)
            tag = status.Get_tag()

            match tag:
                case 1:
                    try:
                        res = f(*args)
                    except:
                        res = "error"
                    mpi_comm.send(res, 0, tag=2)
                case _:
                    done = True


    def get_ready_worker(self):
        for worker in self.workers:
            if worker.is_ready():
                return worker

        return None

class Worker:
    def __init__(self, rank):
        self.rank = rank
        self.is_working = False

    def is_ready(self):
        return not self.is_working

    def set_ready(self):
        self.is_working = False

    def set_working(self):
        self.is_working = True

