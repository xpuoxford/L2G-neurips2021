

import torch.optim as optim

from src.models import *
from src.utils import *
from src.utils_data import *
import argparse
import time
import logging
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%


# parser for hyper-parameters
parser = argparse.ArgumentParser()

# synthetic data:
parser.add_argument('--graph_type', type=str, default='BA', help='{BA, ER, SBM, WS}.')
parser.add_argument('--graph_size', type=int, default=20, help='{20, 50, 100}.')

# model pars:
parser.add_argument('--num_unroll', type=int, default=20, help='Number of Unrolling Layers.')

# training pars:
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train.')
args = parser.parse_args()



#%%

logging.basicConfig(filename='logs/Unrolling_{}_m{}_x{}.log'.format(args.graph_type, args.graph_size, args.num_unroll),
                    filemode='w',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%d-%b-%y %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

#%%

batch_size = 32

data_dir = 'data/dataset_{}_{}nodes.pickle'.format(args.graph_type, args.graph_size)
train_loader, val_loader, test_loader = data_loading(data_dir, batch_size=batch_size)

#%%

net = unrolling(args.num_unroll).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-04)
logging.info(net)

#%%


dur = []

epoch_train_gmse = []
epoch_val_gmse = []

for epoch in range(args.n_epochs):

    train_unrolling_loss, train_gmse, val_gmse = [], [], []

    t0 = time.time()

    net.train()
    for z, w_gt_batch in train_loader:
        z = z.to(device)
        w_gt_batch = w_gt_batch.to(device)
        this_batch_size = w_gt_batch.size()[0]

        optimizer.zero_grad()
        w_list = net.forward(z)

        unrolling_loss = torch.mean(
            torch.stack([acc_loss(w_list[i, :, :], w_gt_batch[i, :], dn=0.9) for i in range(this_batch_size)])
        )

        unrolling_loss.backward()
        optimizer.step()

        gmse = gmse_loss_batch_mean(w_list[:, args.num_unroll-1, :], w_gt_batch)
        train_gmse.append(gmse.item())
        train_unrolling_loss.append(unrolling_loss.item())

    net.eval()
    for z, w_gt_batch in val_loader:
        z = z.to(device)
        w_gt_batch = w_gt_batch.to(device)

        w_list = net(z)
        loss = gmse_loss_batch_mean(w_list[:, args.num_unroll-1, :], w_gt_batch)
        val_gmse.append(loss.item())

    dur.append(time.time() - t0)

    logging.info("Epoch {:04d} | Time(s): {:.4f}".format(epoch + 1, np.mean(dur)))
    logging.info("== unroll loss: {:04.4f} | train gmse : {:04.4f} | val gmse : {:04.4f}>".format(np.mean(train_unrolling_loss),
                                                                                               np.mean(train_gmse),
                                                                                               np.mean(val_gmse)))

    epoch_train_gmse.append(np.mean(train_gmse))
    epoch_val_gmse.append(np.mean(val_gmse))

#%%

save_path = 'saved_model/Unrolling_{}{}_unroll{}.pt'.format(args.graph_type,
                                                            args.graph_size,
                                                            args.num_unroll)

torch.save({'net_state_dict': net.state_dict(),
            'optimiser_state_dict': optimizer.state_dict()
            }, save_path)


logging.info('model saved at: {}'.format(save_path))

#%%

# Test:

for z, w_gt_batch in test_loader:
    test_loss = []

    z = z.to(device)
    w_gt_batch = w_gt_batch.to(device)
    this_batch_size = w_gt_batch.size()[0]

    adj_batch = w_gt_batch.clone()
    adj_batch[adj_batch > 0] = 1

    w_list = net(z)
    w_pred = torch.clamp(w_list[:, args.num_unroll - 1, :], min=0)

    loss_mean = gmse_loss_batch_mean(w_pred, w_gt_batch)
    loss_pred = gmse_loss_batch(w_pred, w_gt_batch)

    layer_loss_batch = torch.stack([layerwise_gmse_loss(w_list[i, :, :], w_gt_batch[i, :]) for i in range(batch_size)])


loss_all_data = loss_pred.detach().cpu().numpy()
final_pred_loss, final_pred_loss_ci, _, _ = mean_confidence_interval(loss_all_data, 0.95)

logging.info(mean_confidence_interval(loss_all_data, 0.95))
aps_auc = binary_metrics_batch(adj_batch, w_pred, device)

layer_loss_mean = [mean_confidence_interval(layer_loss_batch[:,i].detach().cpu().numpy(), confidence=0.95)[0] for i in range(args.num_unroll)]
layer_loss_mean_ci = [mean_confidence_interval(layer_loss_batch[:,i].detach().cpu().numpy(), confidence=0.95)[1] for i in range(args.num_unroll)]

logging.info('layerwise test loss :{}'.format(layer_loss_mean))


#%%


result = {
    'epoch_train_gmse': epoch_train_gmse,
    'epoch_val_gmse': epoch_train_gmse,
    'pred_gmse_mean': final_pred_loss,
    'pred_gmse_mean_ci': final_pred_loss_ci,
    'auc_mean': aps_auc['auc_mean'],
    'auc_ci': aps_auc['auc_ci'],
    'aps_mean': aps_auc['aps_mean'],
    'aps_ci': aps_auc['aps_ci'],
    'layerwise_gmse_mean': layer_loss_mean,
    'layerwise_gmse_mean_ci ': layer_loss_mean_ci
}


#%%
result_path = 'saved_results/Unrolling_{}{}_unroll{}.pt'.format(args.graph_type,
                                                                args.graph_size,
                                                                args.num_unroll)


with open(result_path, 'wb') as handle:
    pickle.dump(result, handle, protocol=4)

logging.info('results saved at: {}'.format(result_path))

#%%
