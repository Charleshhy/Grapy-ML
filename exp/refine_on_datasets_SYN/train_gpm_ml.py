import socket
import timeit
from datetime import datetime
import os
import sys
import glob
import numpy as np
from collections import OrderedDict

sys.path.append('../../')
sys.path.append('../../networks/')
# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import cihp, pascal, atr, pascal_flip, cihp_pascal_atr
from utils import util, get_iou_from_list
from networks import graph, grapy_net
from dataloaders import custom_transforms as tr

#
import argparse
from utils import sampler as sam

gpu_id = 0

nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume


def flip(x, dim):
	indices = [slice(None)] * x.dim()
	indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
								dtype=torch.long, device=x.device)
	return x[tuple(indices)]


def flip_cihp(tail_list):
	'''

	:param tail_list: tail_list size is 1 x n_class x h x w
	:return:
	'''
	# tail_list = tail_list[0]
	tail_list_rev = [None] * 20
	for xx in range(14):
		tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
	tail_list_rev[14] = tail_list[15].unsqueeze(0)
	tail_list_rev[15] = tail_list[14].unsqueeze(0)
	tail_list_rev[16] = tail_list[17].unsqueeze(0)
	tail_list_rev[17] = tail_list[16].unsqueeze(0)
	tail_list_rev[18] = tail_list[19].unsqueeze(0)
	tail_list_rev[19] = tail_list[18].unsqueeze(0)
	return torch.cat(tail_list_rev, dim=0)


def flip_atr(tail_list):
	'''

	:param tail_list: tail_list size is 1 x n_class x h x w
	:return:
	'''
	# tail_list = tail_list[0]
	tail_list_rev = [None] * 18
	for xx in range(9):
		tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
	tail_list_rev[10] = tail_list[9].unsqueeze(0)
	tail_list_rev[9] = tail_list[10].unsqueeze(0)
	tail_list_rev[11] = tail_list[11].unsqueeze(0)
	tail_list_rev[12] = tail_list[13].unsqueeze(0)
	tail_list_rev[13] = tail_list[12].unsqueeze(0)
	tail_list_rev[14] = tail_list[15].unsqueeze(0)
	tail_list_rev[15] = tail_list[14].unsqueeze(0)
	tail_list_rev[16] = tail_list[16].unsqueeze(0)
	tail_list_rev[17] = tail_list[17].unsqueeze(0)

	return torch.cat(tail_list_rev, dim=0)


def get_parser():
	'''argparse begin'''
	parser = argparse.ArgumentParser()

	parser.add_argument('--epochs', default=100, type=int)
	parser.add_argument('--batch', default=16, type=int)
	parser.add_argument('--lr', default=1e-7, type=float)
	parser.add_argument('--numworker', default=12, type=int)
	parser.add_argument('--step', default=10, type=int)
	parser.add_argument('--classes', default=7, type=int)
	parser.add_argument('--testInterval', default=10, type=int)
	parser.add_argument('--loadmodel', default='', type=str, help='load pretrain model')
	parser.add_argument('--hidden_layers', default=128, type=int)
	parser.add_argument('--gpus', default=4, type=int)

	parser.add_argument('--resume_epoch', default=0, type=int)
	parser.add_argument('--hidden_graph_layers', default=256, type=int, help='the hidden layers for both cnn and gcn')

	# Here we first use cihp_pascal_atr, then fine-tune on each dataset
	parser.add_argument('--train_mode', default='cihp_pascal_atr', type=str, help='choose from cihp, '
																				'pascal, atr and cihp_pascal_atr')

	parser.add_argument('--resume_model', default='', type=str, help='resume from the same model')

	parser.add_argument('--beta_main', default='0.8', type=float)
	parser.add_argument('--beta_aux', default='0.2', type=float)

	parser.add_argument('--poly', default='yes', type=str)

	opts = parser.parse_args()
	return opts


def validation(net_, testloader, testloader_flip, epoch, writer, criterion, classes=7, dataset='cihp'):
	running_loss_ts = 0.0
	miou = 0
	# adj1_test, adj2_test, adj3_test, adj4_test, adj5_test, adj6_test = test_graph
	num_img_ts = len(testloader)
	net_.eval()
	pred_list = []
	label_list = []

	with torch.no_grad():
		for ii, sample_batched in enumerate(zip(testloader, testloader_flip)):
			# print(ii)
			inputs, labels_single = sample_batched[0]['image'], sample_batched[0]['label']
			inputs_f, labels_single_f = sample_batched[1]['image'], sample_batched[1]['label']
			inputs = torch.cat((inputs, inputs_f), dim=0)
			labels = torch.cat((labels_single, labels_single_f), dim=0)
			# Forward pass of the mini-batch
			inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)

			with torch.no_grad():
				if gpu_id >= 0:
					inputs, labels, labels_single = inputs.cuda(), labels.cuda(), labels_single.cuda()

			# print(inputs.shape, labels.shape)
			if dataset == 'cihp':
				outputs, outputs_aux = net_.forward((inputs, 0))
				outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
			elif dataset == 'pascal' or dataset == 'cihp_pascal_atr' or dataset == 'cihp_pascal_atr_1_1_1':
				outputs, outputs_aux = net_.forward((inputs, 1))
				outputs = (outputs[0] + flip(outputs[1], dim=-1)) / 2

				# outputs, outputs_aux = net_.forward((inputs, labels, 2))
				# outputs = (outputs[0] + flip(flip_atr(outputs[1]), dim=-1)) / 2

			else:
				outputs, outputs_aux = net_.forward((inputs, 2))
				outputs = (outputs[0] + flip(flip_atr(outputs[1]), dim=-1)) / 2

			outputs = outputs.unsqueeze(0)
			predictions = torch.max(outputs, 1)[1]
			pred_list.append(predictions.cpu())
			label_list.append(labels_single.squeeze(1).cpu())
			loss = criterion(outputs, labels_single, batch_average=True)
			running_loss_ts += loss.item()

			# Print stuff
			if ii % num_img_ts == num_img_ts - 1:
				# if ii == 10:

				miou = get_iou_from_list(pred_list, label_list, n_cls=classes)
				running_loss_ts = running_loss_ts / num_img_ts

				print('Validation:')
				print('[Epoch: %d, numImages: %5d]' % (epoch, ii * 1 + inputs.data.shape[0]))
				writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
				writer.add_scalar('data/test_miour', miou, epoch)
				print('Loss: %f' % running_loss_ts)
				print('MIoU: %f\n' % miou)

				return miou


def main(opts):

	# Some of the settings are not used
	p = OrderedDict()  # Parameters to include in report
	p['trainBatch'] = opts.batch  # Training batch size
	testBatch = 1  # Testing batch size
	useTest = True  # See evolution of the test set when training
	nTestInterval = opts.testInterval  # Run on test set every nTestInterval epochs
	snapshot = 1  # Store a model every snapshot epochs
	p['nAveGrad'] = 1  # Average the gradient of several iterations
	p['lr'] = opts.lr  # Learning rate
	p['lrFtr'] = 1e-5
	p['lraspp'] = 1e-5
	p['lrpro'] = 1e-5
	p['lrdecoder'] = 1e-5
	p['lrother'] = 1e-5
	p['wd'] = 5e-4  # Weight decay
	p['momentum'] = 0.9  # Momentum
	p['epoch_size'] = opts.step  # How many epochs to change learning rate
	p['num_workers'] = opts.numworker
	backbone = 'xception'  # Use xception or resnet as feature extractor,
	nEpochs = opts.epochs

	resume_epoch = opts.resume_epoch

	max_id = 0
	save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
	exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
	runs = glob.glob(os.path.join(save_dir_root, 'run_cihp', 'run_*'))
	for r in runs:
		run_id = int(r.split('_')[-1])
		if run_id >= max_id:
			max_id = run_id + 1
	save_dir = os.path.join(save_dir_root, 'run_cihp', 'run_' + str(max_id))

	print(save_dir)

	# Network definition
	net_ = grapy_net.GrapyMutualLearning(os=16, hidden_layers=opts.hidden_graph_layers)

	modelName = 'deeplabv3plus-' + backbone + '-voc' + datetime.now().strftime('%b%d_%H-%M-%S')
	criterion = util.cross_entropy2d

	log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
	writer = SummaryWriter(log_dir=log_dir)
	writer.add_text('load model', opts.loadmodel, 1)
	writer.add_text('setting', sys.argv[0], 1)

	# Use the following optimizer
	optimizer = optim.SGD(net_.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])

	composed_transforms_tr = transforms.Compose([
		tr.RandomSized_new(512),
		tr.Normalize_xception_tf(),
		tr.ToTensor_()])

	composed_transforms_ts = transforms.Compose([
		tr.Normalize_xception_tf(),
		tr.ToTensor_()])

	composed_transforms_ts_flip = transforms.Compose([
		tr.HorizontalFlip(),
		tr.Normalize_xception_tf(),
		tr.ToTensor_()])

	if opts.train_mode == 'cihp_pascal_atr':
		all_train = cihp_pascal_atr.VOCSegmentation(split='train', transform=composed_transforms_tr, flip=True)
		num_cihp, num_pascal, num_atr = all_train.get_class_num()

		voc_val = atr.VOCSegmentation(split='val', transform=composed_transforms_ts)
		voc_val_flip = atr.VOCSegmentation(split='val', transform=composed_transforms_ts_flip)

		ss = sam.Sampler_uni(num_cihp, num_pascal, num_atr, opts.batch)

		trainloader = DataLoader(all_train, batch_size=p['trainBatch'], shuffle=False, num_workers=18, sampler=ss, drop_last=True)

	elif opts.train_mode == 'cihp_pascal_atr_1_1_1':
		all_train = cihp_pascal_atr.VOCSegmentation(split='train', transform=composed_transforms_tr, flip=True)
		num_cihp, num_pascal, num_atr = all_train.get_class_num()

		voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
		voc_val_flip = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts_flip)

		ss_uni = sam.Sampler_uni(num_cihp, num_pascal, num_atr, opts.batch, balance_id=1)

		trainloader = DataLoader(all_train, batch_size=p['trainBatch'], shuffle=False, num_workers=1, sampler=ss_uni, drop_last=True)

	elif opts.train_mode == 'cihp':
		voc_train = cihp.VOCSegmentation(split='train', transform=composed_transforms_tr, flip=True)
		voc_val = cihp.VOCSegmentation(split='val', transform=composed_transforms_ts)
		voc_val_flip = cihp.VOCSegmentation(split='val', transform=composed_transforms_ts_flip)

		trainloader = DataLoader(voc_train, batch_size=p['trainBatch'], shuffle=True, num_workers=18, drop_last=True)

	elif opts.train_mode == 'pascal':

		# here we train without flip but test with flip
		voc_train = pascal_flip.VOCSegmentation(split='train', transform=composed_transforms_tr)
		voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
		voc_val_flip = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts_flip)

		trainloader = DataLoader(voc_train, batch_size=p['trainBatch'], shuffle=True, num_workers=18, drop_last=True)

	elif opts.train_mode == 'atr':

		# here we train without flip but test with flip
		voc_train = atr.VOCSegmentation(split='train', transform=composed_transforms_tr, flip=True)
		voc_val = atr.VOCSegmentation(split='val', transform=composed_transforms_ts)
		voc_val_flip = atr.VOCSegmentation(split='val', transform=composed_transforms_ts_flip)

		trainloader = DataLoader(voc_train, batch_size=p['trainBatch'], shuffle=True, num_workers=18, drop_last=True)

	else:
		raise NotImplementedError

	if not opts.loadmodel == '':
		x = torch.load(opts.loadmodel)
		net_.load_state_dict_new(x, strict=False)
		print('load model:', opts.loadmodel)
	else:
		print('no model load !!!!!!!!')

	if not opts.resume_model == '':
		x = torch.load(opts.resume_model)
		net_.load_state_dict(x)
		print('resume model:', opts.resume_model)

	else:
		print('we are not resuming from any model')

	# We only validate on pascal dataset to save time
	testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=3)
	testloader_flip = DataLoader(voc_val_flip, batch_size=testBatch, shuffle=False, num_workers=3)

	num_img_tr = len(trainloader)
	num_img_ts = len(testloader)

	# Set the category relations
	c1, c2, p1, p2, a1, a2 = [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]],\
							 [[0], [1, 2, 4, 13], [5, 6, 7, 10, 11, 12], [3, 14, 15], [8, 9, 16, 17, 18, 19]], \
							 [[0], [1, 2, 3, 4, 5, 6]], [[0], [1], [2], [3, 4], [5, 6]], [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]],\
							 [[0], [1, 2, 3, 11], [4, 5, 7, 8, 16, 17], [14, 15], [6, 9, 10, 12, 13]]

	net_.set_category_list(c1, c2, p1, p2, a1, a2)
	if gpu_id >= 0:
		# torch.cuda.set_device(device=gpu_id)
		net_.cuda()

	running_loss_tr = 0.0
	running_loss_ts = 0.0

	running_loss_tr_main = 0.0
	running_loss_tr_aux = 0.0
	aveGrad = 0
	global_step = 0
	miou = 0
	cur_miou = 0
	print("Training Network")

	net = torch.nn.DataParallel(net_)

	# Main Training and Testing Loop
	for epoch in range(resume_epoch, nEpochs):
		start_time = timeit.default_timer()

		if opts.poly:
			if epoch % p['epoch_size'] == p['epoch_size'] - 1:
				lr_ = util.lr_poly(p['lr'], epoch, nEpochs, 0.9)
				optimizer = optim.SGD(net_.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
				writer.add_scalar('data/lr_', lr_, epoch)
				print('(poly lr policy) learning rate: ', lr_)

		net.train()
		for ii, sample_batched in enumerate(trainloader):

			inputs, labels = sample_batched['image'], sample_batched['label']
			# Forward-Backward of the mini-batch
			inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
			global_step += inputs.data.shape[0]

			if gpu_id >= 0:
				inputs, labels = inputs.cuda(), labels.cuda()

			if opts.train_mode == 'cihp_pascal_atr' or opts.train_mode == 'cihp_pascal_atr_1_1_1':
				num_dataset_lbl = sample_batched['pascal'][0].item()

			elif opts.train_mode == 'cihp':
				num_dataset_lbl = 0

			elif opts.train_mode == 'pascal':
				num_dataset_lbl = 1

			else:
				num_dataset_lbl = 2

			outputs, outputs_aux = net.forward((inputs, num_dataset_lbl))

			# print(inputs.shape, labels.shape, outputs.shape, outputs_aux.shape)

			loss_main = criterion(outputs, labels, batch_average=True)
			loss_aux = criterion(outputs_aux, labels, batch_average=True)

			loss = opts.beta_main * loss_main + opts.beta_aux * loss_aux

			running_loss_tr_main += loss_main.item()
			running_loss_tr_aux += loss_aux.item()
			running_loss_tr += loss.item()

			# Print stuff
			if ii % num_img_tr == (num_img_tr - 1):
				running_loss_tr = running_loss_tr / num_img_tr
				running_loss_tr_aux = running_loss_tr_aux / num_img_tr
				running_loss_tr_main = running_loss_tr_main / num_img_tr

				writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)

				writer.add_scalars('data/scalar_group', {'loss': running_loss_tr_main,
														 'loss_aux': running_loss_tr_aux}, epoch)

				print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
				print('Loss: %f' % running_loss_tr)
				running_loss_tr = 0
				stop_time = timeit.default_timer()
				print("Execution time: " + str(stop_time - start_time) + "\n")

			# Backward the averaged gradient
			loss /= p['nAveGrad']
			loss.backward()
			aveGrad += 1

			# Update the weights once in p['nAveGrad'] forward passes
			if aveGrad % p['nAveGrad'] == 0:
				writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)

				if num_dataset_lbl == 0:
					writer.add_scalar('data/total_loss_iter_cihp', loss.item(), global_step)
				if num_dataset_lbl == 1:
					writer.add_scalar('data/total_loss_iter_pascal', loss.item(), global_step)
				if num_dataset_lbl == 2:
					writer.add_scalar('data/total_loss_iter_atr', loss.item(), global_step)

				optimizer.step()
				optimizer.zero_grad()
				aveGrad = 0

			# Show 10 * 3 images results each
			# print(ii, (num_img_tr * 10), (ii % (num_img_tr * 10) == 0))
			if ii % (num_img_tr * 10) == 0:
				grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
				writer.add_image('Image', grid_image, global_step)
				grid_image = make_grid(
					util.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3,
					normalize=False,
					range=(0, 255))
				writer.add_image('Predicted label', grid_image, global_step)
				grid_image = make_grid(
					util.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3,
					normalize=False, range=(0, 255))
				writer.add_image('Groundtruth label', grid_image, global_step)
			print('loss is ', loss.cpu().item(), flush=True)

		# Save the model
		# One testing epoch
		if useTest and epoch % nTestInterval == (nTestInterval - 1):

			cur_miou = validation(net_, testloader=testloader, testloader_flip=testloader_flip, classes=opts.classes,
								epoch=epoch, writer=writer, criterion=criterion, dataset=opts.train_mode)

		torch.cuda.empty_cache()

		if (epoch % snapshot) == snapshot - 1:

			torch.save(net_.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch' + '_current' + '.pth'))
			print("Save model at {}\n".format(
				os.path.join(save_dir, 'models', modelName + str(epoch) + '_epoch-' + str(epoch) + '.pth as our current model')))

			if cur_miou > miou:
				miou = cur_miou
				torch.save(net_.state_dict(), os.path.join(save_dir, 'models', modelName + '_best' + '.pth'))
				print("Save model at {}\n".format(
					os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth as our best model')))

		torch.cuda.empty_cache()


if __name__ == '__main__':
	opts = get_parser()
	main(opts)