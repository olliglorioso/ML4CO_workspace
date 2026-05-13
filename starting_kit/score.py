import os
import json
import torch

def resolve_path(*parts):
    app_base = "/app"
    if os.path.exists(app_base):
        return os.path.join(app_base, *parts)

    local_base = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(local_base, *parts)

PRED_FILE = resolve_path("input", "res", "predictions.pt")
REF_FILE = resolve_path("input", "ref", "graphs.pt")
OUTPUT_FILE = resolve_path("output", "scores.json")

def rel_obj_gap(pred, weights, gt):
    pred_score = (pred.float() * weights).sum()
    return torch.abs((pred_score - gt) / (gt + 1.e-5))


def mvc_check(pred, edge_index):
    return torch.logical_or(pred[edge_index[0]], pred[edge_index[1]]).all().float()


def mis_check(pred, edge_index):
    return (~torch.logical_and(pred[edge_index[0]], pred[edge_index[1]])).all().float()


def mc_check(pred, edge_index):
    return (torch.logical_and(pred[edge_index[0]], pred[edge_index[1]]).sum() == pred.sum() * (pred.sum() - 1)).float()


def main():

    preds = torch.load(PRED_FILE, weights_only=False)
    refs = torch.load(REF_FILE, weights_only=False)

    mis_scores = []
    mvc_scores = []
    mc_scores = []

    mis_feasibilitys = []
    mvc_feasibilitys = []
    mc_feasibilitys = []

    for pred_dict, ref_graph in zip(preds, refs):

        mis_scores.append(rel_obj_gap(pred_dict["mis"], ref_graph.x, ref_graph.mis_obj))
        mvc_scores.append(rel_obj_gap(pred_dict["mvc"], ref_graph.x, ref_graph.mvc_obj))
        mc_scores.append(rel_obj_gap(pred_dict["mc"], ref_graph.x, ref_graph.cli_obj))

        mis_feasibilitys.append(mis_check(pred_dict["mis"], ref_graph.edge_index))
        mvc_feasibilitys.append(mvc_check(pred_dict['mvc'], ref_graph.edge_index))
        mc_feasibilitys.append(mc_check(pred_dict['mc'], ref_graph.edge_index))
    
    mis_obj_gap = (sum(mis_scores) / len(mis_scores)).item()
    mvc_obj_gap = (sum(mvc_scores) / len(mvc_scores)).item()
    mc_obj_gap  = (sum(mc_scores) / len(mc_scores)).item()

    mis_feas = (sum(mis_feasibilitys) / len(mis_feasibilitys)).item()
    mvc_feas = (sum(mvc_feasibilitys) / len(mvc_feasibilitys)).item()
    mc_feas  = (sum(mc_feasibilitys) / len(mc_feasibilitys)).item()

    global_obj_gap = (mis_obj_gap + mvc_obj_gap + mc_obj_gap) / 3.0
    global_feasibility = (mis_feas + mvc_feas + mc_feas) / 3.0

    scores = {
        "global_obj_gap": global_obj_gap,
        "global_feasibility": global_feasibility,
        "MIS_obj_gap": mis_obj_gap,
        "MVC_obj_gap": mvc_obj_gap,
        "MC_obj_gap": mc_obj_gap,
        "MIS_feasibility": mis_feas,
        "MVC_feasibility": mvc_feas,
        "MC_feasibility": mc_feas,
    }

    for key, val in scores.items():
        print(f"Key : {key} - type : {type(val)}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    main()