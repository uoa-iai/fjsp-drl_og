import gym
import env
import PPO_model
import torch
import time
import os
import copy
import torch.nn.functional as F

def get_validate_env(env_paras):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    file_path = "./data_dev/{0}{1}/".format(env_paras["num_jobs"], str.zfill(str(env_paras["num_mas"]),2))
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path+valid_data_files[i]
    env = gym.make('fjsp-v0', case=valid_data_files, env_paras=env_paras, data_source='file')
    return env

def validate(env_paras, env, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = PPO_model.Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    # gantt_result = env.validate_gantt()[0]
    # if not gantt_result:
    #     print("Scheduling Error！！！！！！")
    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)

    tardiness = copy.deepcopy((env.true_tardiness_batch).mean())
    tardiness_batch = copy.deepcopy(env.true_tardiness_batch)

    env.reset()
    print('validating time: ', time.time() - start, '\n')

    return makespan, makespan_batch, tardiness, tardiness_batch


def EED_SPT(env_paras, env):
    '''
      Validate the policy during training, and the process is similar to test
      '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = act_EDD_SPT(state, env_paras, env)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    # gantt_result = env.validate_gantt()[0]
    # if not gantt_result:
    #     print("Scheduling Error！！！！！！")
    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)

    tardiness = copy.deepcopy((env.true_tardiness_batch).mean())
    tardiness_batch = copy.deepcopy(env.true_tardiness_batch)

    env.reset()
    print('validating time 2 : ', time.time() - start, '\n')

    return makespan, makespan_batch, tardiness, tardiness_batch

def act_EDD_SPT(state, env_paras, env):

    remaining_time = state.deadlines_batch[state.batch_idxes, :] - state.time_batch[state.batch_idxes, None]  # todo expand state.time

    scores = torch.ones((len(state.batch_idxes), env.max_jobs, env_paras["num_mas"], ))
    remaining_time_expanded = remaining_time.unsqueeze(-1).expand(-1, -1, env_paras["num_mas"])
    scores = scores / (remaining_time_expanded + 1e-9)
    scores = scores.transpose(1, 2).flatten(1)
    # Matrix indicating whether processing is possible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    eligible_proc = state.ope_ma_adj_batch[state.batch_idxes].gather(1,
                    state.ope_step_batch[..., :, None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[state.batch_idxes])
    # Matrix indicating whether machine is eligible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    ma_eligible = ~state.mask_ma_procing_batch[state.batch_idxes].unsqueeze(1).expand_as(eligible_proc)
    # Matrix indicating whether job is eligible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    job_eligible = ~(state.mask_job_procing_batch[state.batch_idxes] +
                     state.mask_job_finish_batch[state.batch_idxes])[:, :, None].expand_as(eligible_proc)
    # shape: [len(batch_idxes), num_jobs, num_mas]
    eligible = job_eligible & ma_eligible & (eligible_proc == 1)
    mask = eligible.transpose(1, 2).flatten(1)
    scores[~mask] = float('-inf')
    action_probs = F.softmax(scores, dim=1)
    action_indexes = action_probs.argmax(dim=1)
    # Calculate the machine, job and operation index based on the action index
    mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()
    jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()
    opes = state.ope_step_batch[state.batch_idxes, jobs]

    return torch.stack((opes, mas, jobs), dim=1).t()