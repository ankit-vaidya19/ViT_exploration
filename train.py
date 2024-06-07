import argparse
import random
from sys import argv

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.layers.config import set_fused_attn
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet, SVHN, Flowers102
from tqdm import tqdm

parser = argparse.ArgumentParser()
tok_actions = ["cls", "mean", "top_one"]
hook_actions = ["hooks_req", "no_hooks"]
train_type = ["full_ft", "linear_probing"]
parser.add_argument("--dataset", type=str, default="oxford_pets")
parser.add_argument("--data_dir", type=str, default="/mnt/d/ViT")
parser.add_argument("--download_dataset", type=bool, default=True)
parser.add_argument("--save_dir", type=str, default="/mnt/d/ViT")
parser.add_argument("--train_type", choices=train_type, type=str, default="full_ft")
parser.add_argument("--use_hooks", choices=hook_actions, type=str, default="hooks_req")
parser.add_argument("--block", required=(hook_actions[0] in argv), type=int, default=11)
parser.add_argument(
    "--tokens_used",
    required=(hook_actions[0] in argv),
    choices=tok_actions,
    type=str,
    default="mean",
)
parser.add_argument(
    "--num_tokens_for_mean", required=("mean" in argv), type=int, default=10
)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=5e-6)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

set_fused_attn(False, False)


def create_dataloaders(train_transforms, test_transforms):
    if args.dataset == "oxford_pets":
        train_ds = OxfordIIITPet(
            root=args.data_dir,
            split="trainval",
            download=args.download_dataset,
            transform=train_transforms,
        )
        test_ds = OxfordIIITPet(
            root=args.data_dir,
            split="test",
            download=args.download_dataset,
            transform=test_transforms,
        )
        num_classes = len(train_ds.classes)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        return num_classes, train_loader, test_loader
    elif args.dataset == "svhn":
        train_ds = SVHN(
            root=args.data_dir,
            split="train",
            download=args.download_dataset,
            transform=train_transforms,
        )
        test_ds = SVHN(
            root=args.data_dir,
            split="test",
            download=args.download_dataset,
            transform=test_transforms,
        )
        num_classes = len(train_ds.classes)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        return num_classes, train_loader, test_loader
    elif args.dataset == "flowers_102":
        train_ds = Flowers102(
            root=args.data_dir,
            split="train",
            download=args.download_dataset,
            transform=train_transforms,
        )
        test_ds = Flowers102(
            root=args.data_dir,
            split="test",
            download=args.download_dataset,
            transform=test_transforms,
        )
        num_classes = len(train_ds.classes)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        return num_classes, train_loader, test_loader
    else:
        raise ValueError(f"Dataset not found, got {args.dataset=}")


def create_hooks(model):
    hook_dict = {}
    hook_list = []

    def getActivation(name, idx, hook_dict):
        def hook(model, input, output):
            hook_dict[name + str(idx)] = input

        return hook

    hook_list.append(
        model.blocks[args.block - 1].register_forward_hook(
            getActivation("block_op", args.block - 1, hook_dict)
        )
    )
    if args.tokens_used == "mean" or args.tokens_used == "top_one":
        hook_list.append(
            model.blocks[args.block - 1].attn.attn_drop.register_forward_hook(
                getActivation("block_attn", args.block - 1, hook_dict)
            )
        )

    return hook_list, hook_dict


def get_last_lin_input(hook_dict, batch_len):
    if args.tokens_used == "mean":
        attn_map = hook_dict[f"block_attn{args.block-1}"][0].mean(dim=1).sum(dim=-1)
        indices = (
            torch.topk(attn_map, k=args.num_tokens_for_mean, largest=True, sorted=True)
            .indices.unsqueeze(2)
            .expand(batch_len, args.num_tokens_for_mean, args.model_hidden_dim)
        )
        pooler_op = hook_dict[f"block_op{args.block-1}"][0]
        linear_inp = torch.gather(pooler_op, dim=1, index=indices).mean(dim=1)
    elif args.tokens_used == "top_one":
        attn_map = hook_dict[f"block_attn{args.block-1}"][0].mean(dim=1).sum(dim=-1)
        indices = torch.topk(attn_map, k=1, largest=True, sorted=True).indices
        indices = indices.unsqueeze(2).expand(batch_len, 1, 768)
        pooler_op = hook_dict[f"block_op{args.block-1}"][0]
        linear_inp = torch.gather(pooler_op, dim=1, index=indices).squeeze()
    elif args.tokens_used == "cls":
        linear_inp = hook_dict[f"block_op{args.block-1}"][0][:, 0, :]
    return linear_inp


def accuracy(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    acc = np.sum((true == pred).astype(np.float32)) / len(true)
    return acc * 100


def train_with_hooks(model, last_linear, device, train_loader, optim, criterion):
    model.train()
    last_linear.train()
    train_loss = []
    train_preds = []
    train_labels = []
    for batch in tqdm(train_loader):
        hooks, hook_dict = create_hooks(model)
        imgs = batch[0].to(device)
        labels = torch.Tensor(batch[1]).to(device)
        scores = model(imgs)
        for hook in hooks:
            hook.remove()
        linear_inp = get_last_lin_input(hook_dict, len(imgs))
        linear_scores = last_linear(linear_inp)
        loss = criterion(linear_scores, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss.append(loss.detach().cpu().numpy())
        train_labels.append(batch[1])
        train_preds.append(linear_scores.argmax(dim=-1))
    loss = sum(train_loss) / len(train_loss)
    acc = accuracy(
        torch.concat(train_labels, dim=0).cpu(), torch.concat(train_preds, dim=0).cpu()
    )
    print(f"\tTrain\tLoss - {round(loss, 3)}", "\t", f"Accuracy - {round(acc, 3)}")


def train(model, device, train_loader, optim, criterion):
    model.train()
    train_loss = []
    train_preds = []
    train_labels = []
    for batch in tqdm(train_loader):
        imgs = batch[0].to(device)
        labels = torch.Tensor(batch[1]).to(device)
        scores = model(imgs)
        loss = criterion(scores, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss.append(loss.detach().cpu().numpy())
        train_labels.append(batch[1])
        train_preds.append(scores.argmax(dim=-1))
    loss = sum(train_loss) / len(train_loss)
    acc = accuracy(
        torch.concat(train_labels, dim=0).cpu(), torch.concat(train_preds, dim=0).cpu()
    )
    print(f"\tTrain\tLoss - {round(loss, 3)}", "\t", f"Accuracy - {round(acc, 3)}")


def test_with_hooks(model, last_linear, device, test_loader, criterion):
    model.eval()
    last_linear.eval()
    test_loss = []
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            hooks, hook_dict = create_hooks(model)
            imgs = batch[0].to(device)
            labels = torch.Tensor(batch[1]).to(device)
            scores = model(imgs)
            for hook in hooks:
                hook.remove()
            linear_inp = get_last_lin_input(hook_dict, len(imgs))
            linear_scores = last_linear(linear_inp)
            loss = criterion(linear_scores, labels)
            test_loss.append(loss.detach().cpu().numpy())
            test_labels.append(batch[1])
            test_preds.append(linear_scores.argmax(dim=-1))
        loss = sum(test_loss) / len(test_loss)
        acc = accuracy(
            torch.concat(test_labels, dim=0).cpu(),
            torch.concat(test_preds, dim=0).cpu(),
        )
        print(f"\tTest\tLoss - {round(loss, 3)}", "\t", f"Accuracy - {round(acc, 3)}")


def test(model, device, test_loader, criterion):
    model.train()
    test_loss = []
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            imgs = batch[0].to(device)
            labels = torch.Tensor(batch[1]).to(device)
            scores = model(imgs)
            loss = criterion(scores, labels)
            test_loss.append(loss.detach().cpu().numpy())
            test_labels.append(batch[1])
            test_preds.append(scores.argmax(dim=-1))
        loss = sum(test_loss) / len(test_loss)
        acc = accuracy(
            torch.concat(test_labels, dim=0).cpu(),
            torch.concat(test_preds, dim=0).cpu(),
        )
        print(f"\tTest\tLoss - {round(loss, 3)}", "\t", f"Accuracy - {round(acc, 3)}")


def fit(model, last_linear, device, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    if args.use_hooks == "no_hooks":
        optim = torch.optim.Adam(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            train(model, device, train_loader, optim, criterion)
            test(model, device, test_loader, criterion)

        torch.save(
            model.state_dict(),
            f"{args.save_dir}/{args.dataset}_{args.use_hooks}_{args.train_type}.pt",
        )
    else:
        if args.train_type == "linear_probing":
            optim = torch.optim.Adam(
                params=last_linear.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            optim = torch.optim.Adam(
                [{"params": model.parameters()}, {"params": last_linear.parameters()}],
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            train_with_hooks(model, last_linear, device, train_loader, optim, criterion)
            test_with_hooks(model, last_linear, device, test_loader, criterion)
        print("Saving Model")
        torch.save(
            model.state_dict(),
            f"{args.save_dir}/{args.dataset}_{args.use_hooks}_{args.train_type}_{args.tokens_used}_backbone.pt",
        )
        torch.save(
            last_linear.state_dict(),
            f"{args.save_dir}/{args.dataset}_{args.use_hooks}_{args.train_type}_{args.tokens_used}_linear_layer.pt",
        )


def main():

    device = torch.device(args.device if torch.cuda.is_available() else None)

    if device is None:
        raise ValueError("CUDA not found...")

    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    )
    args.model_hidden_dim = model.embed_dim

    data_config = timm.data.resolve_model_data_config(model)
    transforms_train = timm.data.create_transform(**data_config, is_training=True)
    transforms_test = timm.data.create_transform(**data_config, is_training=False)

    num_classes, train_loader, test_loader = create_dataloaders(
        train_transforms=transforms_train, test_transforms=transforms_test
    )

    if args.train_type == "linear_probing":
        for param in model.parameters():
            param.requires_grad = False

    if args.use_hooks == "hooks_req":
        last_linear = nn.Linear(768, num_classes)
        model.to(device)
        last_linear.to(device)
        fit(model, last_linear, device, train_loader, test_loader)

    elif args.use_hooks == "no_hooks":
        last_linear = None
        model.head = nn.Linear(768, num_classes)
        model.to(device)
        fit(model, last_linear, device, train_loader, test_loader)


if __name__ == "__main__":
    main()
