import copy
import json
import os
import random
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

import PPO_model
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env, EED_SPT, act_EDD_SPT


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    setup_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    env_valid = get_validate_env(env_valid_paras)  # Create an environment for validation
    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')
    tardy_best = float('inf')

    # Use visdom to visualize the training process
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_{0}'.format(str_time)
    os.makedirs(save_path)
    # Training curve storage path (average of validation set)
    writer_ave = pd.ExcelWriter('{0}/training_ave_{1}.xlsx'.format(save_path, str_time))
    # Training curve storage path (value of each validating instance)
    writer_100 = pd.ExcelWriter('{0}/training_100_{1}.xlsx'.format(save_path, str_time))

    writer_ave_tardy = pd.ExcelWriter('{0}/training_ave_tardy_{1}.xlsx'.format(save_path, str_time))
    # Training curve storage path (value of each validating instance)
    writer_100_tardy = pd.ExcelWriter('{0}/training_100_tardy_{1}.xlsx'.format(save_path, str_time))

    valid_results = []
    valid_results_100 = []
    valid_results_tardy = []
    valid_results_tardy_100 = []
    st = train_paras["save_timestep"]
    en = train_paras["max_iterations"] + train_paras["save_timestep"]
    data_file = pd.DataFrame(np.arange(st, en, st), columns=["iterations"])
    data_file.to_excel(writer_ave, sheet_name='Sheet1', index=False)
    writer_ave.save()
    writer_ave.close()
    data_file = pd.DataFrame(np.arange(st, en, st), columns=["iterations"])
    data_file.to_excel(writer_100, sheet_name='Sheet1', index=False)
    writer_100.save()
    writer_100.close()

    # Start training iteration
    start_time = time.time()
    env = None
    for i in range(1, train_paras["max_iterations"]+1):
        # Replace training instances every x iteration (x = 20 in paper)
        if (i - 1) % train_paras["parallel_iter"] == 0:
            # \mathcal{B} instances use consistent operations to speed up training
            opes = random.randint(opes_per_job_min, opes_per_job_max)
            nums_ope = [opes for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            env_paras["seed"] = env_paras["seed"] + 1
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("spend_time: ", time.time()-last_time)
        tardiness = copy.deepcopy((env.true_tardiness_batch).mean())
        tardiness_batch = copy.deepcopy(env.true_tardiness_batch)
        if is_viz:
            viz.line(
                X=np.array([i]), Y=np.array([tardiness.item()]),
                win='window{}'.format(4), update='append', name='tardiness', opts=dict(title='Tardiness'))

        env.reset()

        vali_result2, vali_result_1002, vali_result_tardy2, vali_result_tardy_1002 = EED_SPT(env_valid_paras, env)
        if is_viz:
            viz.line(
                X=np.array([i]), Y=np.array([vali_result_tardy2.item()]),
                win='window{}'.format(4), update='append', name='EED Tardiness', opts=dict(title='Tardiness'))
        # Verify the solution
        # gantt_result = env.validate_gantt()[0]
        # if not gantt_result:
        #     print("Scheduling Error！！！！！！")
        # print("Scheduling Finish")
        env.reset()

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            memories.clear_memory()
            if is_viz:
                viz.line(X=np.array([i]), Y=np.array([reward]),
                         win='window{}'.format(0), update='append', name='reward',
                         opts=dict(title='reward of envs'))
                viz.line(X=np.array([i]), Y=np.array([loss]),
                    win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))  # deprecated


        # if iter mod x = 0 then validate the policy (x = 10 in paper)
        if i % train_paras["save_timestep"] == 0:
            print('\nStart validating')
            # Record the average results and the results on each instance
            env_valid.reset()
            vali_result, vali_result_100, vali_result_tardy, vali_result_tardy_100 = validate(env_valid_paras, env_valid, model.policy_old)
            env_valid.reset()
            vali_result2, vali_result_1002, vali_result_tardy2, vali_result_tardy_1002 = EED_SPT(env_valid_paras, env_valid)
            valid_results.append(vali_result.item())
            valid_results_100.append(vali_result_100)
            valid_results_tardy.append(vali_result_tardy.item())
            valid_results_tardy_100.append(vali_result_tardy_100)

            # Save the best model
            if vali_result_tardy < tardy_best:
                tardy_best = vali_result_tardy
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

            if is_viz:
                viz.line(
                    X=np.array([i]), Y=np.array([vali_result.item()]),
                    win='window{}'.format(2), update='append', name='valid', opts=dict(title='makespan of valid'))
                viz.line(
                    X=np.array([i]), Y=np.array([vali_result_tardy.item()]),
                    win='window{}'.format(3), update='append', name='valid', opts=dict(title='tardiness of valid'))
                viz.line(
                    X=np.array([i]), Y=np.array([vali_result2.item()]),
                    win='window{}'.format(2), update='append', name='EED')
                viz.line(
                    X=np.array([i]), Y=np.array([vali_result_tardy2.item()]),
                    win='window{}'.format(3), update='append', name='EED')


    # Save the data of training curve to files
    data = pd.DataFrame(np.array(valid_results).transpose(), columns=["res"])
    data.to_excel(writer_ave, sheet_name='Sheet1', index=False, startcol=1)
    writer_ave.save()
    writer_ave.close()
    cols = train_paras["max_iterations"] // train_paras["save_timestep"]
    column = [i_col for i_col in range(cols)]
    data = pd.DataFrame(np.array(torch.stack(valid_results_100, dim=0).to('cpu')), columns=column)
    data.to_excel(writer_100, sheet_name='Sheet1', index=False, startcol=1)
    writer_100.save()
    writer_100.close()
    data = pd.DataFrame(np.array(valid_results_tardy).transpose(), columns=["res"])
    data.to_excel(writer_ave_tardy, sheet_name='Sheet1', index=False, startcol=1)
    writer_ave_tardy.save()
    writer_ave_tardy.close()
    column = [i_col for i_col in range(len(valid_results_tardy_100))]
    data = pd.DataFrame(np.array(torch.stack(valid_results_tardy_100, dim=0).to('cpu')), columns=column)
    data.to_excel(writer_100_tardy, sheet_name='Sheet1', index=False, startcol=1)
    writer_100_tardy.save()
    writer_100_tardy.close()

    print("total_time: ", time.time() - start_time)

if __name__ == '__main__':
    main()