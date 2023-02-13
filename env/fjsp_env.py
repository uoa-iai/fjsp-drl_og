import sys
import gym
import torch

from dataclasses import dataclass
from env.load_data import load_fjs, nums_detec, load_for_l, num_ma_detec, load_fjs_new
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import copy
from utils.my_utils import read_json, write_json
import torch.nn.functional as F


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch: torch.Tensor = None

    deadlines_batch: torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time,
               deadlines_batch,
               cal_cumul_adj_batch, ope_pre_adj_batch, ope_sub_adj_batch, opes_appertain_batch, end_ope_biases_batch,
               nums_ope_batch, nums_ope_batch_dynamic, num_ope_biases_batch):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time

        self.deadlines_batch = deadlines_batch
        self.cal_cumul_adj_batch = cal_cumul_adj_batch
        self.ope_pre_adj_batch = ope_pre_adj_batch
        self.ope_sub_adj_batch = ope_sub_adj_batch
        self.opes_appertain_batch = opes_appertain_batch
        self.end_ope_biases_batch = end_ope_biases_batch
        self.nums_ope_batch = nums_ope_batch
        self.nums_ope_batch_dynamic = nums_ope_batch_dynamic
        self.num_ope_biases_batch = num_ope_biases_batch


def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    '''
    Convert job features into operation features (such as dimension)
    '''
    return feat_job_batch.gather(1, opes_appertain_batch)


class FJSPEnv(gym.Env):
    '''
    FJSP environment
    '''

    def __init__(self, case, env_paras, data_source='case'):
        '''
        :param case: The instance generator or the addresses of the instances
        :param env_paras: A dictionary of parameters for the environment
        :param data_source: Indicates that the instances came from a generator or files
        '''

        # load paras
        # static
        self.show_mode = env_paras["show_mode"]  # Result display mode (deprecated in the final experiment)
        self.batch_size = env_paras["batch_size"]  # Number of parallel instances during training
        self.num_jobs = env_paras["num_jobs"]  # Number of jobs
        self.num_mas = env_paras["num_mas"]  # Number of machines
        self.paras = env_paras  # Parameters
        self.device = env_paras["device"]  # Computing device for PyTorch
        seed_val = env_paras["seed"]  # seed for repeatable values
        self.ma_util = env_paras["ma_util"]  # set machine utilisation
        self.inital_jobs = env_paras["init_jobs"]  # initial number of jobs at the start of simulation
        self.tot_jobs = env_paras["tot_jobs"]  # total number of jobs to be completed
        # load instance
        self.num_data = 10  # The amount of data extracted from instance
        tensors = [[] for _ in range(self.num_data)]
        self.num_opes = 0  # max num of opes in system
        self.max_jobs = 0  # max num of jobs in system
        self.num_opes_system = torch.zeros(self.batch_size)
        self.num_jobs_system = torch.zeros(self.batch_size)
        self.tot_jobs_added = [0 for _ in range(self.batch_size)]  # keeps track of no. jobs added for each case
        self.tot_ops_added = torch.zeros(self.batch_size) # keeps track of no. ops added for each case
        self.arrival_times = [[] for _ in range(self.batch_size)]  # arrival times for upcoming jobs
        self.arrival_rates = []  # arrival rates
        self.ss_jobs = [0 for _ in range(self.batch_size)]
        self.library = []  # keeps track of job library of each instance

        # set seeds
        self.rng_job_idx = np.random.default_rng(seed_val)
        self.rng_job_arr = np.random.default_rng(seed_val)
        self.rng_ddt = np.random.default_rng(seed_val)

        added_jobs = [[] for _ in range(self.batch_size)]

        if data_source == 'case':  # Generate instances through generators
            arrival_rate = case.get_arrival_rate()  # get arrival rate of case generator
            self.arrival_rates = [arrival_rate for _ in range(self.batch_size)]

            for i in range(self.batch_size):
                self.library.append(case.get_case(i)[0])  # Generate an instance and save it
                self.initialise_arrival_times(i)
                added_jobs[i].append(self.library[i][0])
                while self.arrival_times[i][0] <= 0:
                    job_idx = self.rng_job_idx.integers(1, self.num_jobs)
                    added_jobs[i].append(self.library[i][job_idx])
                    self.arrival_times[i].pop(0)
                num_jobs, num_mas, num_opes = nums_detec(added_jobs[i])
                # Records the maximum number of operations in the parallel instances
                self.num_opes_system[i] += num_opes
                self.num_jobs_system[i] += num_jobs
                self.tot_jobs_added[i] += num_jobs
                self.tot_ops_added[i] += num_opes
                self.next_arrival_times(i)
        else:  # Load instances from files
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.readlines()
                    self.library.append(line)
                added_jobs[i].append(self.library[i][0])
                service_rate = load_for_l(self.library[i])
                num_mas = num_ma_detec(added_jobs[i])
                arrival_rate = self.ma_util * num_mas * service_rate
                self.arrival_rates.append(arrival_rate)
                self.initialise_arrival_times(i)
                while self.arrival_times[i][0] <= 0:
                    job_idx = self.rng_job_idx.integers(1, self.num_jobs)
                    added_jobs[i].append(self.library[i][job_idx])
                    self.arrival_times[i].pop(0)
                num_jobs, num_mas, num_opes = nums_detec(added_jobs[i])
                # Records the maximum number of operations in the parallel instances
                self.num_opes_system[i] += num_opes
                self.num_jobs_system[i] += num_jobs
                self.tot_jobs_added[i] += num_jobs
                self.tot_ops_added[i] += num_opes
                self.next_arrival_times(i)

        # Records the maximum number of operations in the parallel instances
        self.num_opes = int(max(self.num_opes_system))
        self.max_jobs = int(max(self.num_jobs_system))

        # load feats
        for i in range(self.batch_size):
            load_data = load_fjs(added_jobs[i], self.num_mas, self.num_opes, self.max_jobs)
            for j in range(self.num_data-1):
                tensors[j].append(load_data[j])

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        self.nums_ope_batch_dynamic = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)
        # shape: (batch_size, num_jobs), the deadline for each job
        self.deadlines_batch = torch.stack(tensors[8], dim=0).long()

        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.time = torch.zeros(self.batch_size)  # Current time of the environment
        self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch < 0, self.num_opes - 1,
                                                self.end_ope_biases_batch)
        self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch >= self.num_opes, self.num_opes - 1,
                                                self.end_ope_biases_batch)
        self.ope_step_batch = torch.where(self.ope_step_batch < 0, self.num_opes - 1, self.ope_step_batch)
        self.ope_step_batch = torch.where(self.ope_step_batch >= self.num_opes, self.num_opes - 1, self.ope_step_batch)
        self.opes_appertain_batch = torch.where(self.opes_appertain_batch < 0, 0, self.opes_appertain_batch)
        '''
        features, dynamic
            ope:
                Status
                Number of neighboring machines
                Processing time
                Number of unscheduled operations in the job
                Job completion time
                Start time
                Job Deadline
            ma:
                Number of neighboring operations
                Available time
                Utilization
        '''
        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], self.num_mas))

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 6, :] = convert_feat_job_2_ope(self.deadlines_batch, self.opes_appertain_batch)
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                 fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                fill_value=False)
        '''
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
        '''
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]
        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        '''
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.tardiness_batch = torch.sum(
            (self.feat_opes_batch[:, 4, :] - self.feat_opes_batch[:, 6, :]).gather(1, self.end_ope_biases_batch),
            dim=1) / self.num_jobs
        self.true_tardiness_batch = torch.zeros(self.batch_size)
        # print(f'finish time {self.feat_opes_batch[:, 4, :]}, deadline {self.feat_opes_batch[:, 6, :]},
        # self.tardiness {self.tardiness_batch}')
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        for i in self.batch_idxes:
            self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0
            self.feat_opes_batch[i, 0, int(self.num_opes_system[i]):] = 1
            self.mask_job_procing_batch[i, int(self.num_jobs_system[i]):] = True
            self.mask_job_finish_batch[i, int(self.num_jobs_system[i]):] = True

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)

        # Save initial data for reset - only includes dynamic features
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)

        self.old_num_opes_system = copy.deepcopy(self.num_opes_system)
        self.old_num_jobs_system = copy.deepcopy(self.num_jobs_system)

        self.old_tot_jobs_added = copy.deepcopy(self.tot_jobs_added)
        self.old_tot_ops_added = copy.deepcopy(self.tot_ops_added)
        self.old_arrival_times = copy.deepcopy(self.arrival_times)

        self.old_ope_pre_adj_batch = copy.deepcopy(self.ope_pre_adj_batch)
        self.old_ope_sub_adj_batch = copy.deepcopy(self.ope_sub_adj_batch)
        self.old_opes_appertain_batch = copy.deepcopy(self.opes_appertain_batch)

        self.old_ope_step_batch = copy.deepcopy(self.ope_step_batch)
        self.old_end_ope_biases_batch = copy.deepcopy(self.end_ope_biases_batch)
        self.old_nums_ope_batch = copy.deepcopy(self.nums_ope_batch)
        self.old_nums_ope_batch_dynamic = copy.deepcopy(self.nums_ope_batch_dynamic)
        self.old_deadlines_batch = copy.deepcopy(self.deadlines_batch)
        self.old_num_ope_biases_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.old_mask_job_procing_batch = copy.deepcopy(self.mask_job_procing_batch)
        self.old_mask_job_finish_batch = copy.deepcopy(self.mask_job_finish_batch)

        # get rng states
        self.old_rng_job_idx_state = self.rng_job_idx.__getstate__()
        self.old_rng_job_arr_state = self.rng_job_arr.__getstate__()
        self.old_rng_ddt_state = self.rng_ddt.__getstate__()

    def step(self, actions):
        '''
        Environment transition function
        '''
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.N += 1
        #print(f'tot jobs {self.tot_jobs_added}')

        # print(f'len opes {len(opes)}')
        # print(f'opes {opes}, mas {mas}, jobs {jobs}, tim {self.time}')
        # print(f'feat {self.feat_opes_batch[:, 0, :]}')
        #print(f'self.ope_step_batch {self.ope_step_batch}')
        #print(f'self.end {self.end_ope_biases_batch}')
        # print(f'self.start {self.num_ope_biases_batch}')
        #print(f'machines_batch {self.machines_batch}')

        j = 0
        for i in self.batch_idxes:
            # if (self.batch_size > 20) and (i == 14):
            #     print(f'opes {opes[i]}, mas {mas[i]}, jobs {jobs[i]}, tim {self.time[i]}')
            #     if (self.time[i] == 99.0):
            #         print('heyo')
            #     print(f'feat {self.feat_opes_batch[i, 0, :]}')
            if self.feat_opes_batch[i, 0, opes[j]] == 1:
                print(f'i {i}')
                print(f'Scheduling already scheduled operation')
                print(f'opes {opes[i]}')
                print(f'feats {self.feat_opes_batch[i, 0, :]}')
                raise Exception
            j += 1

        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             proc_times), dim=1)
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1,
                                opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0

        # Update 'Number of unscheduled operations in the job'
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        self.nums_ope_batch_dynamic[self.batch_idxes, jobs] -= 1
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i] + 1] -= 1

        # Update 'Start time' and 'Job completion time'
        self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled  # real start time of scheduled opes
        un_scheduled = 1 - is_scheduled  # unscheduled opes
        unsched_starts = torch.where((self.feat_opes_batch[:,5,:] <= self.time.unsqueeze(-1)) & (un_scheduled == 1), 1, 0)
        print(f'unsched_starts {unsched_starts}')
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                                   self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() \
                         * un_scheduled  # estimate start time of unscheduled opes
        start_ope = self.ope_step_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 5, start_ope[i]:end_ope[i] + 1] -= 1
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] +
                          self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[
                                                                                  self.batch_idxes, :])
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[
                                                                                              self.batch_idxes, :])
        # todo leave appertain as was... manually adjust value and set to zero.
        for i in self.batch_idxes:
            self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0
            self.feat_opes_batch[i, 0, int(self.num_opes_system[i]):] = 1

        # Update partial schedule (state)
        # self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas),
        #                                                                dim=1)
        # self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        # self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + \
        #                                                self.feat_opes_batch[self.batch_idxes, 2, :]
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))
        self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

        # Update feature vectors of machines
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :],
                                                                          dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, mas] = self.time[self.batch_idxes] + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz

        # Update other variable according to actions
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()


        maxy = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        # tardy = torch.sum((self.feat_opes_batch[:, 4, :] - self.feat_opes_batch[:, 6, :]).
        # gather(1, self.end_ope_biases_batch), dim=1)/self.num_jobs_system
        tardy, _ = torch.max(
            (self.feat_opes_batch[:, 4, :] - self.feat_opes_batch[:, 6, :]).gather(1, self.end_ope_biases_batch), dim=1)
        # self.reward_batch = self.makespan_batch - max
        self.reward_batch = self.tardiness_batch - tardy
        self.makespan_batch = self.time
        self.tardiness_batch = tardy

        cloned_ope_ma_adj_batch = self.ope_ma_adj_batch.clone()
        cloned_proc_times_batch = self.proc_times_batch.clone()
        cloned_cal_cumul_adj_batch = self.cal_cumul_adj_batch.clone()
        cloned_ope_pre_adj_batch = self.ope_pre_adj_batch.clone()
        cloned_ope_sub_adj_batch = self.ope_sub_adj_batch.clone()
        cloned_opes_appertain_batch = self.opes_appertain_batch.clone()
        cloned_feat_opes_batch = self.feat_opes_batch.clone()

        cloned_ope_step_batch = self.ope_step_batch.clone()
        cloned_end_ope_biases_batch = self.end_ope_biases_batch.clone()
        cloned_nums_ope_batch = self.nums_ope_batch.clone()
        cloned_nums_ope_batch_dynamic = self.nums_ope_batch_dynamic.clone()
        cloned_deadlines_batch = self.deadlines_batch.clone()
        cloned_num_ope_biases_batch = self.num_ope_biases_batch.clone()
        cloned_mask_job_procing_batch = self.mask_job_procing_batch.clone()
        cloned_mask_job_finish_batch = self.mask_job_finish_batch.clone()

        # remove any completed jobs
        j = 0
        for i in self.batch_idxes:
            job_idx = int(jobs[j])
            mas_idx = int(mas[j])
            j += 1
            #print(f'jobs {jobs} {job_idx} complete')
            if self.ope_step_batch[i, job_idx] == self.end_ope_biases_batch[i, job_idx] + 1:
                # print(f'JOB COMPLETE {job_idx} bi {i}')
                start_ope = int(self.num_ope_biases_batch[i, job_idx])  # index of starting op
                end_ope = int(self.end_ope_biases_batch[i, job_idx])  # index of ending
                num_rmd = int(self.feat_opes_batch.shape[2]) - end_ope - 1  # no. of elems. b/w end and last op in system
                # print(f'st {start_ope}, en {end_ope}, rm {num_rmd}, num {self.num_opes}')
                self.num_jobs_system[i] -= 1
                self.num_opes_system[i] -= (end_ope - start_ope + 1)

                self.true_tardiness_batch[i] += (
                        -self.feat_opes_batch[i, 6, end_ope] + self.feat_opes_batch[i, 4, end_ope])

                # print(f'self.ope_ma_adj_batch {self.ope_ma_adj_batch}')
                # print(f'self.proc_times_batch {self.proc_times_batch}')
                self.ope_ma_adj_batch[i, start_ope:start_ope + num_rmd, :] = cloned_ope_ma_adj_batch[i, end_ope + 1:, :]
                self.ope_ma_adj_batch[i, int(self.num_opes_system[i]):, :] = 0
                self.proc_times_batch[i, start_ope:start_ope + num_rmd, :] = cloned_proc_times_batch[i, end_ope + 1:, :]
                self.proc_times_batch[i, int(self.num_opes_system[i]):, :] = 0
                # print(f'self.ope_ma_adj_batch {self.ope_ma_adj_batch}')
                # print(f'self.proc_times_batch {self.proc_times_batch}')

                # print(f'self.cal_cumul_adj_batch {self.cal_cumul_adj_batch}')
                # print(f'self.ope_pre_adj_batch {self.ope_pre_adj_batch}')
                # print(f'self.ope_sub_adj_batch {self.ope_sub_adj_batch}')
                self.cal_cumul_adj_batch[i, start_ope:start_ope + num_rmd, start_ope:start_ope + num_rmd] = \
                    cloned_cal_cumul_adj_batch[i, end_ope + 1:, end_ope + 1:]
                self.cal_cumul_adj_batch[i, int(self.num_opes_system[i]):, int(self.num_opes_system[i]):] = 0
                self.ope_pre_adj_batch[i, start_ope:start_ope + num_rmd, :] = \
                    cloned_ope_pre_adj_batch[i, end_ope + 1:, :]
                self.ope_pre_adj_batch[i, int(self.num_opes_system[i]):, :] = 0
                self.ope_sub_adj_batch[i, start_ope:start_ope + num_rmd, :] = \
                    cloned_ope_sub_adj_batch[i, end_ope + 1:, :]
                self.ope_sub_adj_batch[i, int(self.num_opes_system[i]):, :] = 0
                # print(f'a3 self.cal_cumul_adj_batch {self.cal_cumul_adj_batch}')
                # print(f'self.ope_pre_adj_batch {self.ope_pre_adj_batch}')
                # print(f'self.ope_sub_adj_batch {self.ope_sub_adj_batch}')

                # print(f'apper {self.opes_appertain_batch}')
                self.opes_appertain_batch[i, start_ope:start_ope + num_rmd] = \
                    cloned_opes_appertain_batch[i, end_ope + 1:]
                self.opes_appertain_batch[i, start_ope:] -= 1
                # self.opes_appertain_batch[i, int(self.num_opes_system[i]):] = int(self.num_jobs_system[i])
                self.opes_appertain_batch[i, int(self.num_opes_system[i]):] = 0
                # print(f'apper {self.opes_appertain_batch}')

                # print(f'feat opes {self.feat_opes_batch}')
                # print(f'b4 2 feat {self.feat_opes_batch[:, 5, :]}')
                self.feat_opes_batch[i, :, start_ope:start_ope + num_rmd] = cloned_feat_opes_batch[i, :, end_ope + 1:]
                self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0
                self.feat_opes_batch[i, 0, int(self.num_opes_system[i]):] = 1

                # print(f'a3 2 feat {self.feat_opes_batch[:, 5, :]}')
                # print(f'feat opes {self.feat_opes_batch}')

                # update job tensors
                # print(f'num_jos {num_jobs}')
                # print(f'self.ope_step_batch {self.ope_step_batch}')
                self.ope_step_batch[i, job_idx:self.max_jobs - 1] = cloned_ope_step_batch[i, job_idx + 1:]
                self.ope_step_batch[i, job_idx:] -= (end_ope - start_ope + 1)
                self.ope_step_batch[i, int(self.num_jobs_system[i]):] = -1
                # print(f'self.ope_step_batch {self.ope_step_batch}')

                # print(f'self.end_ope_biases_batch {self.end_ope_biases_batch}')
                self.end_ope_biases_batch[i, job_idx:self.max_jobs - 1] = cloned_end_ope_biases_batch[i, job_idx + 1:]
                self.end_ope_biases_batch[i, job_idx:] -= (end_ope - start_ope + 1)
                self.end_ope_biases_batch[i, int(self.num_jobs_system[i]):] = -1
                # print(f'self.end_ope_biases_batch {self.end_ope_biases_batch}')

                # print(f'self.nums_ope_batch {self.nums_ope_batch}')
                self.nums_ope_batch[i, job_idx:self.max_jobs - 1] = cloned_nums_ope_batch[i, job_idx + 1:]
                self.nums_ope_batch[i, self.max_jobs - 1] = 0
                self.nums_ope_batch_dynamic[i, job_idx:self.max_jobs - 1] = cloned_nums_ope_batch_dynamic[i, job_idx + 1:]
                self.nums_ope_batch_dynamic[i, self.max_jobs - 1] = 0
                # print(f'self.nums_ope_batch {self.nums_ope_batch}')

                # print(f'self.deadlines_batch {self.deadlines_batch}')
                self.deadlines_batch[i, job_idx:self.max_jobs - 1] = cloned_deadlines_batch[i, job_idx + 1:]
                self.deadlines_batch[i, self.max_jobs - 1] = 0
                # print(f'self.deadlines_batch {self.deadlines_batch}')

                # print(f'self.num_ope_biases_batch {self.num_ope_biases_batch}')
                self.num_ope_biases_batch[i, job_idx:self.max_jobs - 1] = cloned_num_ope_biases_batch[i, job_idx + 1:]
                self.num_ope_biases_batch[i, job_idx:] -= (end_ope - start_ope + 1)
                self.num_ope_biases_batch[i, int(self.num_jobs_system[i]):] = -1
                # print(f'self.num_ope_biases_batch {self.num_ope_biases_batch}')

                self.mask_job_procing_batch[i, job_idx:self.max_jobs - 1] = \
                    cloned_mask_job_procing_batch[i, job_idx + 1:]
                self.mask_job_procing_batch[i, self.max_jobs - 1] = True

                # print(f'self.mask_job_finish_batch {self.mask_job_finish_batch}')
                self.mask_job_finish_batch[i, job_idx:self.max_jobs - 1] = cloned_mask_job_finish_batch[i, job_idx + 1:]
                # print(f'self.mask_job_finish_batch {self.mask_job_finish_batch}')
                # print(f'num {self.num_jobs}')
                self.mask_job_finish_batch[i, self.max_jobs - 1] = True
                # print(f'self.mask_job_finish_batch {self.mask_job_finish_batch}')

                job_ids_adjusted = torch.where(self.machines_batch[i, :, 3] > job_idx, self.machines_batch[i, :, 3] - 1,
                                               self.machines_batch[i, :, 3])
                self.machines_batch[i, :, 3] = job_ids_adjusted
                self.machines_batch[i, mas_idx, 3] = -1

        self.max_jobs = int(torch.max(self.num_jobs_system))
        self.num_opes = int(torch.max(self.num_opes_system))
        if (len(self.ope_step_batch[0]) > self.max_jobs):  # resize to max number of jobs in system
            self.ope_step_batch = self.ope_step_batch[:, :self.max_jobs]
            self.end_ope_biases_batch = self.end_ope_biases_batch[:, :self.max_jobs]
            self.nums_ope_batch = self.nums_ope_batch[:, :self.max_jobs]
            self.nums_ope_batch_dynamic = self.nums_ope_batch_dynamic[:, :self.max_jobs]
            self.deadlines_batch = self.deadlines_batch[:, :self.max_jobs]
            self.mask_job_procing_batch = self.mask_job_procing_batch[:, :self.max_jobs]
            self.mask_job_finish_batch = self.mask_job_finish_batch[:, :self.max_jobs]
            self.num_ope_biases_batch = self.num_ope_biases_batch[:, :self.max_jobs]
        if (self.ope_ma_adj_batch.size(dim=1) > self.num_opes):  # resize to max number of ops in system
            self.ope_ma_adj_batch = self.ope_ma_adj_batch[:, :self.num_opes, :]
            self.proc_times_batch = self.proc_times_batch[:, :self.num_opes, :]
            self.cal_cumul_adj_batch = self.cal_cumul_adj_batch[:, :self.num_opes, :self.num_opes]
            self.ope_pre_adj_batch = self.ope_pre_adj_batch[:, :self.num_opes, :self.num_opes]
            self.ope_sub_adj_batch = self.ope_sub_adj_batch[:, :self.num_opes, :self.num_opes]
            self.opes_appertain_batch = self.opes_appertain_batch[:, :self.num_opes]
            self.feat_opes_batch = self.feat_opes_batch[:, :, :self.num_opes]
            self.schedules_batch = self.schedules_batch[:, :self.num_opes, :]

        self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch < 0, self.num_opes - 1,
                                                self.end_ope_biases_batch)
        self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch >= self.num_opes, self.num_opes - 1,
                                                self.end_ope_biases_batch)
        self.ope_step_batch = torch.where(self.ope_step_batch < 0, self.num_opes - 1, self.ope_step_batch)
        self.ope_step_batch = torch.where(self.ope_step_batch >= self.num_opes, self.num_opes - 1, self.ope_step_batch)

        self.opes_appertain_batch = torch.where(self.opes_appertain_batch < 0, 0, self.opes_appertain_batch)

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        flag_trans_2_next_time = self.if_no_eligible()
        count = 0
        tot_jobs_added = torch.FloatTensor(self.tot_jobs_added)
        while ~((~(((flag_trans_2_next_time == 0) & (~self.done_batch)) | ((self.done_batch) & (tot_jobs_added < self.tot_jobs)))).all()):
            self.next_time(flag_trans_2_next_time)
            self.add_job()
            flag_trans_2_next_time = self.if_no_eligible()
            tot_jobs_added = torch.FloatTensor(self.tot_jobs_added)
            count += 1
            self.done_batch = self.mask_job_finish_batch.all(dim=1)
            self.done = self.done_batch.all()
            if count > 10:
                print('infinite loop')
                flag_need_trans = (flag_trans_2_next_time == 0) & (~self.done_batch)
                print(f'flag {flag_need_trans}')
                print(f'tot jobs {tot_jobs_added}')
                raise Exception

        # Update the vector for uncompleted instances
        mask_finish = (self.N + 1) <= self.tot_ops_added
        # print(f'masky {mask_finish}')
        if ~(mask_finish.all()):
            # print(f'mask finish {mask_finish} N {self.N}, {self.nums_opes}')
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        # Update state of the environment
        # print(f'self.done_batch {self.done_batch}')
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch, self.proc_times_batch,
                          self.ope_ma_adj_batch, self.mask_job_procing_batch, self.mask_job_finish_batch,
                          self.mask_ma_procing_batch,
                          self.ope_step_batch, self.time, self.deadlines_batch, self.cal_cumul_adj_batch,
                          self.ope_pre_adj_batch,
                          self.ope_sub_adj_batch, self.opes_appertain_batch, self.end_ope_biases_batch,
                          self.nums_ope_batch, self.nums_ope_batch_dynamic,
                          self.num_ope_biases_batch)
        return self.state, self.reward_batch, self.done_batch

    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        # ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
        #                              self.end_ope_biases_batch, self.ope_step_batch)
        ope_step_batch = self.ope_step_batch
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                           self.proc_times_batch.size(
                                                                                               2)))

        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(
            torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
            dim=[1, 2])
        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        '''
        # need to transit
        tot_jobs_added = torch.FloatTensor(self.tot_jobs_added)
        flag_need_trans = ((flag_trans_2_next_time == 0) & (~self.done_batch)) | (self.done_batch & (tot_jobs_added < self.tot_jobs))
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        # remain available_time greater than current time
        # b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, 4, :]) + 1.0)
        b = torch.where(a > self.time[:, None], a, torch.max(a) + 1.0)
        # b1 = torch.where(b < torch.max(a), torch.max(a), b)
        # Return the minimum value of available_time (the time to transit to)
        c = torch.min(b, dim=1)[0]
        # check go to next job arrival time
        arrival_times = [self.arrival_times[i][0] for i in range(self.batch_size)]
        arrival_times = torch.FloatTensor(arrival_times)

        c_2 = torch.where(((arrival_times < c) & (tot_jobs_added < self.tot_jobs)) | ((self.done_batch) & (tot_jobs_added < self.tot_jobs)), arrival_times, c)

        # Detect the machines that completed (at above time)
        d = torch.where((a <= c_2[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)

        # The time for each batch to transit to or stay in
        e = torch.where(flag_need_trans, c_2, self.time)
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)

        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]
        #print(f'time {self.time} bi {batch_idxes} job {job_idxes}')

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        #print(f'mask {self.mask_job_procing_batch}')

        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

    def initialise_arrival_times(self, batch_idx):
        tim = 0
        # todo
        # if self.batch_size == 1:
        #     ss = self.L
        # else:
        #     ss = self.L[batch_idx]
        for _ in range(self.inital_jobs):
            self.arrival_times[batch_idx].append(int(tim))
        for _ in range(3):
            inter_arrival_time = self.rng_job_arr.exponential(1 / self.arrival_rates[batch_idx])
            tim += inter_arrival_time
            self.arrival_times[batch_idx].append(int(tim))
            while self.arrival_times[batch_idx][-1] == self.arrival_times[batch_idx][-2]:
                tim = self.arrival_times[batch_idx][-1] + self.rng_job_arr.exponential(
                    1 / self.arrival_rates[batch_idx])
                self.arrival_times[batch_idx].append(int(tim))

    def next_arrival_times(self, batch_idx):
        tim = self.arrival_times[batch_idx][-1] + self.rng_job_arr.exponential(1 / self.arrival_rates[batch_idx])
        self.arrival_times[batch_idx].append(int(tim))
        while self.arrival_times[batch_idx][-1] == self.arrival_times[batch_idx][-2]:
            tim = self.arrival_times[batch_idx][-1] + self.rng_job_arr.exponential(1 / self.arrival_rates[batch_idx])
            self.arrival_times[batch_idx].append(int(tim))

    def add_job(self):

        # detect which batches have additions of jobs
        # add jobs to "added jobs" list
        # generate next arrival times
        added_jobs = [[] for _ in range(self.batch_size)]
        tensors = [[] for _ in range(self.num_data)]
        og_ops_added = copy.deepcopy(self.num_opes_system).int()
        og_jobs_added = copy.deepcopy(self.num_jobs_system).int()
        og_max_opes = copy.deepcopy(self.num_opes)
        og_max_jobs = copy.deepcopy(self.max_jobs)
        for i in range(self.batch_size):
            if self.tot_jobs_added[i] >= self.tot_jobs:
                # print(f'enough jobs added {i}')
                pass
            else:
                added_jobs[i].append(self.library[i][0])
                while (self.arrival_times[i][0] <= self.time[i]) and (self.tot_jobs_added[i] < self.tot_jobs):  # if job to be added
                    job_idx = self.rng_job_idx.integers(1, self.num_jobs)
                    added_jobs[i].append(self.library[i][job_idx])  # append jobs to be added in one tim step
                    self.arrival_times[i].pop(0)
                    self.tot_jobs_added[i] += 1
                    # print(f'JOB ADDED {job_idx} bi {i}')
                if len(added_jobs[i]) > 0:
                    added_num_jobs, num_mas, added_num_opes = nums_detec(added_jobs[i])
                    self.num_opes_system[i] += added_num_opes
                    self.num_jobs_system[i] += added_num_jobs
                    self.tot_ops_added[i] += added_num_opes

                    self.next_arrival_times(i)

        if not any(len(added_job) > 1 for added_job in added_jobs):  # if no jobs were added return
            return False
        # Records the maximum number of operations in the parallel instances
        self.num_opes = int(max(max(self.num_opes_system), self.num_opes))
        self.max_jobs = int(max(max(self.num_jobs_system), self.max_jobs))

        pad_size_jobs = self.max_jobs - og_max_jobs
        padding = (0, pad_size_jobs)
        self.ope_step_batch = F.pad(input=self.ope_step_batch, pad=padding, value=-1)
        pad_size = self.max_jobs - og_max_jobs
        padding = (0, pad_size)
        self.mask_job_procing_batch = F.pad(input=self.mask_job_procing_batch, pad=padding, value=True)
        self.mask_job_finish_batch = F.pad(input=self.mask_job_finish_batch, pad=padding, value=True)
        pad_size_opes = self.num_opes - int(og_max_opes)
        padding = (0, pad_size_opes)
        self.feat_opes_batch = F.pad(input=self.feat_opes_batch, pad=padding, value=1)
        for i in range(self.batch_size):
            # if len(added_jobs[i]) > 1:
            load_data = load_fjs_new(added_jobs[i],
                                     self.num_mas,
                                     self.num_opes,
                                     self.max_jobs,
                                     og_jobs_added[i],
                                     self.nums_ope_batch[i],
                                     self.nums_ope_batch_dynamic[i],
                                     og_ops_added[i],
                                     self.deadlines_batch[i],
                                     self.opes_appertain_batch[i],
                                     self.num_ope_biases_batch[i],
                                     self.proc_times_batch[i],
                                     self.ope_pre_adj_batch[i],
                                     self.cal_cumul_adj_batch[i], )
            hmm = load_data[5]
            self.ope_step_batch[i:i+1, int(og_jobs_added[i]):int(self.max_jobs)] = hmm[int(og_jobs_added[i]):int(self.max_jobs)]
            self.mask_job_procing_batch[i, int(og_jobs_added[i]):int(self.num_jobs_system[i])] = False
            self.mask_job_finish_batch[i, int(og_jobs_added[i]):int(self.num_jobs_system[i])] = False
            self.feat_opes_batch[i, 0, int(og_ops_added[i]):int(self.num_opes_system[i])] = 0
            for j in range(self.num_data):
                tensors[j].append(load_data[j])


        # stack tensors
        # update features

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[8], dim=0).float()

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        self.nums_ope_batch_dynamic = torch.stack(tensors[7], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)
        # shape: (batch_size, num_jobs), the deadline for each job
        self.deadlines_batch = torch.stack(tensors[9], dim=0).long()

        self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch < 0, self.num_opes - 1,
                                                self.end_ope_biases_batch)
        self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch >= self.num_opes, self.num_opes - 1,
                                                self.end_ope_biases_batch)
        self.ope_step_batch = torch.where(self.ope_step_batch < 0, self.num_opes - 1, self.ope_step_batch)
        self.ope_step_batch = torch.where(self.ope_step_batch >= self.num_opes, self.num_opes - 1, self.ope_step_batch)
        self.opes_appertain_batch = torch.where(self.opes_appertain_batch < 0, 0, self.opes_appertain_batch)

        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))

        feat_opes_batch[:, 0, :] = self.feat_opes_batch[:, 0, :]
        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch_dynamic, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        for i in range(self.batch_size):
            feat_opes_batch[:, 5, :og_ops_added[i]] = self.feat_opes_batch[:, 5, :og_ops_added[i]]
            # feat_opes_batch[:, 5, og_ops_added[i]:int(self.num_opes_system[i])] += self.time[i]
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 6, :] = convert_feat_job_2_ope(self.deadlines_batch, self.opes_appertain_batch)
        self.feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        self.feat_opes_batch = feat_opes_batch

        for i in self.batch_idxes:
            self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0
            self.feat_opes_batch[i, 0, int(self.num_opes_system[i]):] = 1
            self.mask_job_procing_batch[i, int(self.num_jobs_system[i]):] = True
            self.mask_job_finish_batch[i, int(self.num_jobs_system[i]):] = True

    def reset(self):
        '''
        Reset the environment to its initial state
        '''
        self.rng_job_idx.__setstate__(self.old_rng_job_idx_state)
        self.rng_job_arr.__setstate__(self.old_rng_job_arr_state)
        self.rng_ddt.__setstate__(self.old_rng_ddt_state)

        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)

        self.ope_pre_adj_batch = copy.deepcopy(self.old_ope_pre_adj_batch)
        self.ope_sub_adj_batch = copy.deepcopy(self.old_ope_sub_adj_batch)
        self.opes_appertain_batch = copy.deepcopy(self.old_opes_appertain_batch)

        self.end_ope_biases_batch = copy.deepcopy(self.old_end_ope_biases_batch)
        self.nums_ope_batch = copy.deepcopy(self.old_nums_ope_batch)
        self.nums_ope_batch_dynamic = copy.deepcopy(self.old_nums_ope_batch_dynamic)
        self.deadlines_batch = copy.deepcopy(self.old_deadlines_batch)
        self.num_ope_biases_batch = copy.deepcopy(self.old_num_ope_biases_batch)

        self.num_opes_system = copy.deepcopy(self.old_num_opes_system)
        self.num_jobs_system = copy.deepcopy(self.old_num_jobs_system)

        self.num_opes = int(max(self.num_opes_system))
        self.max_jobs = int(max(self.num_jobs_system))

        self.tot_jobs_added = copy.deepcopy(self.old_tot_jobs_added)
        self.tot_ops_added = copy.deepcopy(self.old_tot_ops_added)
        self.arrival_times = copy.deepcopy(self.old_arrival_times)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                 fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.tardiness_batch = \
            torch.sum(
                (self.feat_opes_batch[:, 4, :] - self.feat_opes_batch[:, 6, :]).gather(1, self.end_ope_biases_batch),
                dim=1)[
                0] / self.num_jobs
        self.true_tardiness_batch = torch.zeros(self.batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)

        for i in self.batch_idxes:
            self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0
            self.feat_opes_batch[i, 0, int(self.num_opes_system[i]):] = 1
            self.mask_job_procing_batch[i, int(self.num_jobs_system[i]):] = True
            self.mask_job_finish_batch[i, int(self.num_jobs_system[i]):] = True

        return self.state

    def render(self, mode='human'):
        '''
        Deprecated in the final experiment
        '''
        if self.show_mode == 'draw':
            num_jobs = self.num_jobs
            num_mas = self.num_mas
            print(sys.argv[0])
            color = read_json("./utils/color_config")["gantt_color"]
            if len(color) < num_jobs:
                num_append_color = num_jobs - len(color)
                color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for _ in
                          range(num_append_color)]
            write_json({"gantt_color": color}, "./utils/color_config")
            for batch_id in range(self.batch_size):
                schedules = self.schedules_batch[batch_id].to('cpu')
                fig = plt.figure(figsize=(10, 6))
                fig.canvas.set_window_title('Visual_gantt')
                axes = fig.add_axes([0.1, 0.1, 0.72, 0.8])
                y_ticks = []
                y_ticks_loc = []
                for i in range(num_mas):
                    y_ticks.append('Machine {0}'.format(i))
                    y_ticks_loc.insert(0, i + 1)
                labels = [''] * num_jobs
                for j in range(num_jobs):
                    labels[j] = "job {0}".format(j + 1)
                patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(self.num_jobs)]
                axes.cla()
                axes.set_title(u'FJSP Schedule')
                axes.grid(linestyle='-.', color='gray', alpha=0.2)
                axes.set_xlabel('Time')
                axes.set_ylabel('Machine')
                axes.set_yticks(y_ticks_loc, y_ticks)
                axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
                axes.set_ybound(1 - 1 / num_mas, num_mas + 1 / num_mas)
                for i in range(int(self.nums_opes[batch_id])):
                    id_ope = i
                    idx_job, idx_ope = self.get_idx(id_ope, batch_id)
                    id_machine = schedules[id_ope][1]
                    axes.barh(id_machine,
                              0.2,
                              left=schedules[id_ope][2],
                              color='#b2b2b2',
                              height=0.5)
                    axes.barh(id_machine,
                              schedules[id_ope][3] - schedules[id_ope][2] - 0.2,
                              left=schedules[id_ope][2] + 0.2,
                              color=color[idx_job],
                              height=0.5)
                plt.show()
        return

    def get_idx(self, id_ope, batch_id):
        '''
        Get job and operation (relative) index based on instance index and operation (absolute) index
        '''
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def validate_gantt(self):
        '''
        Verify whether the schedule is feasible
        '''
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i]) - 1):
                        break
                    if ma_gantt[i][j][2] > ma_gantt[i][j + 1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2] - ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1

        # Check job order and overlap
        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i] + j]
                    step_next = schedule[num_ope_biases[i] + j + 1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        # Check whether there are unscheduled operations
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0] == 1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch

    def close(self):
        pass
