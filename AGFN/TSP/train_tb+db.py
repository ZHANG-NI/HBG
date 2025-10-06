import math
import os
import random
import time
import copy
from tqdm import tqdm
import numpy as np
import torch

from net_tbdb import Net
from search import search_
from utils import gen_pyg_data, load_val_dataset

import wandb
#python train_tsp.py 200

EPS = 1e-10
T = 5  
START_NODE = None  

def forward_calculate(paths_raw, heu_mat, device):
    path_lengths = torch.tensor([len(p) for p in paths_raw], device=device)
    max_path_len = path_lengths.max().item() - 1  

    padded_paths = torch.full((len(paths_raw), max_path_len + 1), -1, device=device)
    for i, path in enumerate(paths_raw):
        padded_paths[i, : len(path)] = torch.tensor(path, device=device)

    start_nodes = padded_paths[:, :-1]  
    end_nodes = padded_paths[:, 1:]    

    valid_mask = (start_nodes != -1) & (end_nodes != -1)
    heu_mat = heu_mat.clone().fill_diagonal_(1)
    path_edge_probs = torch.zeros_like(start_nodes, dtype=torch.float32, device=device)
    path_edge_probs[valid_mask] = heu_mat[start_nodes[valid_mask], end_nodes[valid_mask]]
    path_edge_probs[valid_mask]= torch.log(path_edge_probs[valid_mask])
    path_edge_probs1 = torch.cumsum(path_edge_probs, dim=1)
    return path_edge_probs, path_edge_probs1-path_edge_probs

def Z_calculate(paths, x, device):
    u = paths.permute(1, 0) # shape: (n_ants, max_seq_len)
    Z = x[u[:, :-1]]
    Z_b = x[u[:, 1:]]
    Z = torch.cumsum(Z, dim=1)
    col_indices = torch.arange(1, Z.shape[1] + 1, device=Z.device, dtype=torch.float32).unsqueeze(0).T
    Z = Z / col_indices
    col_indices = torch.arange(2, Z.shape[1] + 2, device=Z.device, dtype=torch.float32).unsqueeze(0).T
    Z_b = Z_b / col_indices
    return Z.squeeze(-1), Z_b.squeeze(-1)


def gen_costs(distances, paths):
    u = paths.permute(1, 0) # shape: (n_ants, max_seq_len)
    v = torch.roll(u, shifts=-1, dims=1) 
    reward = distances[u[:, :-1], v[:, :-1]]
    #print(u,v,reward)
    reward = torch.cumsum(reward, dim=1)
    #col_indices = torch.arange(1, reward.shape[1] + 1, device=reward.device, dtype=torch.float32)
    #reward = reward / col_indices
    return reward


def des_F(
        model,
        model_D,
        optimizer,
        optimizer_D,
        dataset,
        generate,
        alpha=1.0,
    ):
    model.eval()
    model_D.train()
    num = 0
    loss_all_D = torch.tensor([0.0], device=DEVICE)
    for model_inp, distances in dataset:
        her, _, x = model(model_inp, True)
        mat = model.reshape(model_inp, her) + EPS
        her_D, _, _ = model_D(model_inp, True)
        mat_D = model_D.reshape(model_inp, her_D) + EPS
        search = search_(distances, generate, heuristic=mat,heuristic_target=mat_D.to(DEVICE), device=DEVICE, local_search='nls')
        _, _, _,score = search.generate_route(alpha=alpha, start_node=START_NODE)

        R=torch.exp(score)
        R=R.mean(0)
        loss_D1=torch.pow(R,2)#torch.pow(R, 2)
        
        loss_all_D+=loss_D1.mean()
        num += 1
    loss_D=loss_all_D/ num
    optimizer_D.zero_grad()
    loss_D.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model_D.parameters(), max_norm=3.0, norm_type=2)  # type: ignore
    optimizer_D.step()


def des_T(
        model,
        model_D,
        optimizer,
        optimizer_D,
        dataset,
        generate,
        alpha=1.0,
    ):
    model.eval()
    model_D.train()
    num = 0
    loss_all_D = torch.tensor([0.0], device=DEVICE)
    for pyg_data, distances in dataset:
        her, _, x = model(pyg_data, return_logZ=True)
        mat = model.reshape(pyg_data, her) + EPS
        her_D, _, _ = model_D(pyg_data, return_logZ=True)
        mat_D = model_D.reshape(pyg_data, her_D) + EPS
        search = search_(distances, generate, heuristic=mat,heuristic_target=mat_D.to(DEVICE), device=DEVICE)
        _, _, paths, _ = search.generate_route(alpha=alpha, start_node=START_NODE)
        _, _,score = search.improve_route(paths, start_node=START_NODE)
        R=torch.exp(score)
        R=R.mean(0)
        #print(R)
        loss_D=torch.pow(1-R,2)
        loss_all_D+=loss_D.mean()
        num += 1

    loss_D=loss_all_D/ num
    optimizer_D.zero_grad()
    loss_D.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model_D.parameters(), max_norm=3.0, norm_type=2)  # type: ignore
    optimizer_D.step()

def generate_train(
        model,
        model_D,
        optimizer,
        optimizer_D,
        dataset,
        generate,
        alpha=1.0,
        beta=100.0,
        it=0
    ):
    model.train()
    model_D.eval()
    ##################################################
    # wandb
    _train_mean_cost = 0.0
    _train_min_cost = 0.0
    _train_mean_cost_nls = 0.0
    _train_min_cost_nls = 0.0
    _train_entropy = 0.0
    _logZ_mean = torch.tensor(0.0, device=DEVICE)
    _logZ_nls_mean = torch.tensor(0.0, device=DEVICE)
    ##################################################
    loss_all = torch.tensor(0.0, device=DEVICE)
    num = 0

    for model_inp, distances in dataset:
        her, flows, x = model(model_inp, True)
        mat = model.reshape(model_inp, her) + EPS
        flow = flows[0]
        her_D, _, _ = model_D(model_inp, True)
        mat_D = model_D.reshape(model_inp, her_D) + EPS
        search = search_(distances, generate, heuristic=mat,heuristic_target=mat_D.to(DEVICE), device=DEVICE)
        cos, forward, paths, forward_D = search.generate_route(alpha=alpha, start_node=START_NODE)
        adv = cos - cos.mean() 

        R=torch.exp(forward_D)
        R=R.mean(0)
        #print(R.mean())
        forward_flow = forward.sum(0) + flow.expand(generate)  # type: ignore
        backward_flow = math.log(1 / (2 * model_inp.x.shape[0])) - adv.detach() * beta-(1-R)* beta
        tb_loss = torch.pow(forward_flow - backward_flow, 2).mean()
        loss_all += tb_loss

        forward, forward_pro = forward_calculate(paths.T, mat, DEVICE)
        Z, Z_b = Z_calculate(paths, x, DEVICE)
        reward = gen_costs(distances, paths)
        reward_f = torch.roll(reward, shifts=1, dims=1)
        reward_f[:, 0] = 0
        reward = reward - reward.mean(0)
        reward_f = reward_f - reward_f.mean(0)
        forward_flow = forward + Z- (reward_f.detach())* beta#forward + flow + log_b_f_matrix + forward_pro#forward + Z#forward + (reward.detach())* beta + Z#logZ #Z
        backward_flow = 0 - (reward.detach())* beta+ Z_b#back_pro + (reward_f.detach())* beta +Z_b
        db_loss = torch.pow(forward_flow - backward_flow, 2).mean(-1).mean(0)
        #print(tb_loss)
        loss_all += db_loss
        num += 1
        ##################################################
        # wandb
        if USE_WANDB:
            _train_mean_cost += cos.mean().item()
            _train_min_cost += cos.min().item()

            normed_heumat = mat / mat.sum(dim=1, keepdim=True)
            entropy = -(normed_heumat * torch.log(normed_heumat)).sum(dim=1).mean()
            _train_entropy += entropy.item()

            _logZ_mean += flow
        ##################################################


    loss_all = loss_all / num
    loss = loss_all

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3.0, norm_type=2)  # type: ignore
    optimizer.step()
    count=num
    ##################################################
    # wandb
    if USE_WANDB:
        wandb.log(
            {
                "train_mean_cost": _train_mean_cost / count,
                "train_min_cost": _train_min_cost / count,
                "train_entropy": _train_entropy / count,
                "train_loss": loss.item(),
                "logZ": _logZ_mean.item() / count,
                "beta": beta,
            },
            step=it,
        )
    ##################################################


def validate(model, model_inp, distances, generate):
    model.eval()
    her = model(model_inp)
    mat = model.reshape(model_inp, her) + EPS

    search = search_(
        distances,
        generate,
        heuristic=mat,
        device='cuda:0',
    )

    costs = search.generate_route(inference=True, start_node=START_NODE)[0]
    bas = costs.mean().item()
    bes_cost = costs.min().item()
    bes1, div1 = search.val(n_iterations=1, inference=True, start_node=START_NODE)
    besT, divT = search.val(n_iterations=T - 1, inference=True, start_node=START_NODE)
    return np.array([bas, bes_cost, bes1, besT, div1, divT])


def get_data(num, scale, k_sparse):
    for _ in range(num):
        instance = torch.rand(size=(scale, 2), device=DEVICE)
        yield gen_pyg_data(instance, k_sparse, start_node=START_NODE)


def train_20step(
    scale,
    sparse,
    generate,
    process,
    iter,
    net,
    net_D,
    optimizer,
    optimizer_D,
    bats = 1,
    alpha=1.0,
    beta=100.0,
):
    for i in tqdm(range(iter), desc="Train"):
        it = (process - 1) * iter + i
        data = get_data(bats, scale, sparse)
        if process%5!=1:
            print("1")
            generate_train(net,net_D, optimizer, optimizer_D, data, generate, alpha, beta, it)
        else:
            if random.uniform(0,1)<0.7:
                print("2")
                des_F(net,net_D, optimizer, optimizer_D, data, generate, alpha)
            else:
                print("3")
                des_T(net,net_D, optimizer, optimizer_D, data, generate, alpha)


@torch.no_grad()
def validation(dataset, generate, net, process):
    result = []
    for data, distances in tqdm(dataset, desc="Val"):
        result.append(validate(net, data, distances, generate))
    avg_stats = [i.item() for i in np.stack(result).mean(0)]
    ##################################################
    print(f"epoch {process}:", avg_stats)
    # wandb
    if USE_WANDB:
        wandb.log(
            {
                "val_baseline": avg_stats[0],
                "val_best_sample_cost": avg_stats[1],
                "val_best_aco_1": avg_stats[2],
                "val_best_aco_T": avg_stats[3],
                "val_diversity_1": avg_stats[4],
                "val_diversity_T": avg_stats[5],
                "epoch": process,
            },
            step=process * 20,
        )
    ##################################################

    return avg_stats[3]


def main(
        scale,
        sparse,
        generate,
        vgenerate,
        iter,
        process,
        lr=1e-4,
        bats=3,
        vals=None,
        viter=5,
        path="../pretrained/tsp_nls",
        run_name="",
        alpha_schedule_params=(0.8, 1.0, 5),  # (alpha_min, alpha_max, alpha_flat_epochs)
        beta_schedule_params=(50, 500, 5),  # (beta_min, beta_max, beta_flat_epochs)
    ):
    path = os.path.join(path, str(scale), run_name)
    os.makedirs(path, exist_ok=True)

    net = Net(gfn=True, Z_out_dim=1, start_node=START_NODE).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, process, eta_min=lr * 0.1)
    net_D=Net(gfn=True, Z_out_dim=1).to(DEVICE)
    optimizer_D = torch.optim.AdamW(net_D.parameters(), lr=lr)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, process, eta_min=lr * 0.1)

    valdata = load_val_dataset(scale, sparse, DEVICE, start_node=START_NODE)
    valdata = valdata[:(vals or len(valdata))]
    besr = validation(valdata, vgenerate, net, 0)
    time_total = 0
    for epoch in range(1, process + 1):

        alpha_min, alpha_max, alpha_flat_epochs = alpha_schedule_params
        alpha = alpha_min + (alpha_max - alpha_min) * min((epoch - 1) / (process - alpha_flat_epochs), 1.0)

        # Beta Schedule
        beta_min, beta_max, beta_flat_epochs = beta_schedule_params
        beta = beta_min + (beta_max - beta_min) * min(math.log(epoch) / math.log(process - beta_flat_epochs), 1.0)

        begin = time.time()
        train_20step(
            scale,
            sparse,
            generate,
            epoch,
            iter,
            net,
            net_D,
            optimizer,
            optimizer_D,
            bats,
            alpha,
            beta,
        )
        time_total += time.time() - begin

        if epoch % viter == 0:
            curr = validation(valdata, vgenerate, net, epoch)
            if curr < besr:
                torch.save(net.state_dict(), os.path.join(path, f"best.pt"))
                besr = curr
            torch.save(net.state_dict(), os.path.join(path, f"{epoch}.pt"))

        scheduler.step()
        scheduler_D.step()
    print('\ntotal training duration:', time_total)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", metavar='N', type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-l", "--lr", metavar='η', type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-p", "--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("-a", "--generate", type=int, default=10, help="Number of generate")
    parser.add_argument("-va", "--vgenerate", type=int, default=100, help="Number of generate for validation")
    parser.add_argument("-b", "--bats", type=int, default=10, help="Batch size")#5
    parser.add_argument("-s", "--steps", type=int, default=20, help="Steps per epoch")#20
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Epochs to run")
    parser.add_argument("-v", "--vals", type=int, default=10, help="Number of instances for validation")#10
    parser.add_argument("-o", "--output", type=str, default="../pretrained/tsp_AGFN_edge",
                        help="The directory to store checkpoints")
    parser.add_argument("--viter", type=int, default=1, help="The interval to validate model")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default="", help="Run name")
    ### alpha
    parser.add_argument("--alpha_min", type=float, default=0.8, help='Inverse temperature min')
    parser.add_argument("--alpha_max", type=float, default=1.0, help='Inverse temperature max')
    parser.add_argument("--alpha_flat_epochs", type=int, default=5, help='Inverse temperature flat epochs')

    parser.add_argument("--beta_min", type=float, default=None, help='Beta min')
    parser.add_argument("--beta_max", type=float, default=None, help='Beta max')
    parser.add_argument("--beta_flat_epochs", type=int, default=5, help='Beta flat epochs')


    args = parser.parse_args()

    args.k_sparse = args.nodes // 5

    beta_min_map = {100:200, 200: 200, 500: 200, 1000: 200 if args.pretrained is None else 1000}
    args.beta_min = beta_min_map[args.nodes]
    beta_max_map = {100:1000, 200: 1000, 500: 1000, 1000: 1000}
    args.beta_max = beta_max_map[args.nodes]

    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    USE_WANDB = not args.disable_wandb

    # seed everything
    torch.manual_seed(1000)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    ##################################################
    # wandb
    run_name = f"[{args.run_name}]" if args.run_name else ""
    run_name += f"tsp{args.nodes}"
    pretrained_name = (
        args.pretrained.replace("../pretrained/tsp_AGFN_edge/", "").replace("/", "_").replace(".pt", "")
        if args.pretrained is not None else None
    )
    run_name += f"{'' if pretrained_name is None else '_fromckpt-'+pretrained_name}"
    if USE_WANDB:
        #wandb.login(key="741075a8c05db693d155c2341a8b5855f9e4620f")
        wandb.init(project="gfacs-tsp_nls", name=run_name)
        wandb.config.update(args)
        wandb.config.update({"T": T, "model": "GFACS"})
    ##################################################

    main(
        args.nodes,
        args.k_sparse,
        args.generate,
        args.vgenerate,
        args.steps,
        args.epochs,
        lr=args.lr,
        bats=args.bats,
        vals=args.vals,
        viter=args.viter,
        path=args.output,
        run_name=run_name,
        alpha_schedule_params=(args.alpha_min, args.alpha_max, args.alpha_flat_epochs),
        beta_schedule_params=(args.beta_min, args.beta_max, args.beta_flat_epochs),
    )