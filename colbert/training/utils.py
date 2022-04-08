import os
import torch

# from colbert.utils.runs import Run
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS
from colbert.infra.run import Run


def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)

# change the "manage_checkpoints" to "manage_checkpoints_consumed_all_triples" as we use it aftere consumed akk triples
def manage_checkpoints_consumed_all_triples(args, colbert, optimizer, batch_idx, savepath=None, consumed_all_triples=False):
    arguments = dict(args)

    # TODO: Call provenance() on the values that support it??

    checkpoints_path = savepath or os.path.join(Run().path_, 'checkpoints')
    name = None

    try:
        save = colbert.save
    except:
        save = colbert.module.save

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    
    path_save = None

    if consumed_all_triples or (batch_idx % 2000 == 0):
        # name = os.path.join(path, "colbert.dnn")
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
        path_save = os.path.join(checkpoints_path, "colbert")

    if batch_idx in SAVED_CHECKPOINTS:
        # name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
        path_save = os.path.join(checkpoints_path, f"colbert-{batch_idx}")

    if path_save:
        print(f"#> Saving a checkpoint to {path_save} ..")

        checkpoint = {}
        checkpoint['batch'] = batch_idx
        # checkpoint['epoch'] = 0
        # checkpoint['model_state_dict'] = model.state_dict()
        # checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['arguments'] = arguments

        save(path_save)

    return path_save

# Add this function to supoort treaining loop logic
def manage_checkpoints(args, colbert, optimizer, amp, batch_idx, num_per_epoch, epoch_idx=0, train_loss=0):
    arguments = args.input_arguments.__dict__

    saved_name = ""
    # path = os.path.join(Run.path, 'checkpoints')
    # V2 uses "from colbert.infra.run import Run"  vs. V1 uses "from colbert.utils.runs import Run"
    path = os.path.join(Run().path_, 'checkpoints')

    if not os.path.exists(path):
        os.makedirs(path)
    prefix = os.path.join(path, "colbert.dnn")

    if args.save_epochs == -1:
        if batch_idx % args.save_steps == 0:
            saved_name = prefix + f".batch_{batch_idx}.model"
            # save_checkpoint(saved_name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, arguments)
            # save_checkpoint(saved_name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, args.model_type)
            save_checkpoint(saved_name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, args.model_type, arguments)
    else:
        if batch_idx * args.bsize * args.nranks % int(args.save_epochs * num_per_epoch) < args.bsize * args.nranks:
            if args.save_epochs.is_integer():
                saved_name = prefix + f".epoch_{epoch_idx}.model"
            else:
                saved_name = prefix + f".epoch_{epoch_idx}_batch_{batch_idx}.model"

            # save_checkpoint(saved_name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, arguments)
            # save_checkpoint(saved_name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, args.model_type)
            save_checkpoint(saved_name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, args.model_type, arguments)


    if batch_idx in SAVED_CHECKPOINTS or batch_idx == args.maxsteps:
        name = prefix + f".batch_{batch_idx}.model"
        if not name == saved_name:
            # save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, arguments)
            # save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, args.model_type)
            save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, args.model_type, arguments)

    if (batch_idx * args.bsize * args.nranks) % (args.epochs * num_per_epoch) < args.bsize * args.nranks:
        name = prefix + f".epoch_{args.epochs - 1}.model"
        if not name == saved_name:
            # save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, arguments)
            # save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, args.model_type)
            save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, args.model_type, arguments)
