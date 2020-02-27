import timeit
import numpy as np
from PIL import Image
import os
import sys
sys.path.append('../../')
# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2

from dataloaders import cihp, atr, pascal
from networks import graph, grapy_net
from dataloaders import custom_transforms as tr

#
import argparse
import copy
import torch.nn.functional as F
from test_from_disk import eval_, eval_with_numpy


gpu_id = 1

label_colours = [(0,0,0)
				, (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
				, (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]


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

def decode_labels(mask, num_images=1, num_classes=20):
	"""Decode batch of segmentation masks.

	Args:
	  mask: result of inference after taking argmax.
	  num_images: number of images to decode from the batch.
	  num_classes: number of classes to predict (including background).

	Returns:
	  A batch with num_images RGB images of the same size as the input.
	"""
	n, h, w = mask.shape
	assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
	outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
	for i in range(num_images):
	  img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
	  pixels = img.load()
	  for j_, j in enumerate(mask[i, :, :]):
		  for k_, k in enumerate(j):
			  if k < num_classes:
				  pixels[k_,j_] = label_colours[k]
	  outputs[i] = np.array(img)
	return outputs

def get_parser():
	'''argparse begin'''
	parser = argparse.ArgumentParser()
	LookupChoices = type('', (argparse.Action,), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

	parser.add_argument('--epochs', default=100, type=int)
	parser.add_argument('--batch', default=16, type=int)
	parser.add_argument('--lr', default=1e-7, type=float)
	parser.add_argument('--numworker', default=12, type=int)
	parser.add_argument('--step', default=30, type=int)
	# parser.add_argument('--loadmodel',default=None,type=str)
	parser.add_argument('--classes', default=7, type=int)
	parser.add_argument('--testepoch', default=10, type=int)
	parser.add_argument('--loadmodel', default='', type=str)
	parser.add_argument('--txt_file', default='', type=str)
	parser.add_argument('--hidden_layers', default=128, type=int)
	parser.add_argument('--gpus', default=4, type=int)
	parser.add_argument('--output_path', default='./results/', type=str)
	parser.add_argument('--gt_path', default='./results/', type=str)

	parser.add_argument('--resume_model', default='', type=str)

	parser.add_argument('--hidden_graph_layers', default=256, type=int)
	parser.add_argument('--dataset', default='cihp', type=str)

	opts = parser.parse_args()
	return opts


def main(opts):

	with open(opts.txt_file, 'r') as f:
		img_list = f.readlines()

	net = grapy_net.GrapyMutualLearning(os=16, hidden_layers=opts.hidden_graph_layers)

	if gpu_id >= 0:
		net.cuda()

	if not opts.resume_model == '':
		x = torch.load(opts.resume_model)
		net.load_state_dict(x)

		print('resume model:', opts.resume_model)

	else:
		print('we are not resuming from any model')

	if opts.dataset == 'cihp':
		val = cihp.VOCSegmentation
		val_flip = cihp.VOCSegmentation

		vis_dir = '/cihp_output_vis/'
		mat_dir = '/cihp_output/'

		num_dataset_lbl = 0

	elif opts.dataset == 'pascal':

		val = pascal.VOCSegmentation
		val_flip = pascal.VOCSegmentation

		vis_dir = '/pascal_output_vis/'
		mat_dir = '/pascal_output/'

		num_dataset_lbl = 1

	elif opts.dataset == 'atr':
		val = atr.VOCSegmentation
		val_flip = atr.VOCSegmentation

		vis_dir = '/atr_output_vis/'
		mat_dir = '/atr_output/'

		print("atr_num")
		num_dataset_lbl = 2

	## multi scale
	scale_list=[1,0.5,0.75,1.25,1.5,1.75]
	testloader_list = []
	testloader_flip_list = []
	for pv in scale_list:
		composed_transforms_ts = transforms.Compose([
			tr.Scale_(pv),
			tr.Normalize_xception_tf(),
			tr.ToTensor_()])

		composed_transforms_ts_flip = transforms.Compose([
			tr.Scale_(pv),
			tr.HorizontalFlip(),
			tr.Normalize_xception_tf(),
			tr.ToTensor_()])

		voc_val = val(split='val', transform=composed_transforms_ts)
		voc_val_f = val_flip(split='val', transform=composed_transforms_ts_flip)

		testloader = DataLoader(voc_val, batch_size=1, shuffle=False, num_workers=4)
		testloader_flip = DataLoader(voc_val_f, batch_size=1, shuffle=False, num_workers=4)

		testloader_list.append(copy.deepcopy(testloader))
		testloader_flip_list.append(copy.deepcopy(testloader_flip))

	print("Eval Network")

	if not os.path.exists(opts.output_path + vis_dir):
		os.makedirs(opts.output_path + vis_dir)
	if not os.path.exists(opts.output_path + mat_dir):
		os.makedirs(opts.output_path + mat_dir)

	start_time = timeit.default_timer()
	# One testing epoch
	total_iou = 0.0

	c1, c2, p1, p2, a1, a2 = [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]],\
							 [[0], [1, 2, 4, 13], [5, 6, 7, 10, 11, 12], [3, 14, 15], [8, 9, 16, 17, 18, 19]], \
							 [[0], [1, 2, 3, 4, 5, 6]], [[0], [1], [2], [3, 4], [5, 6]], [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]],\
							 [[0], [1, 2, 3, 11], [4, 5, 7, 8, 16, 17], [14, 15], [6, 9, 10, 12, 13]]

	net.set_category_list(c1, c2, p1, p2, a1, a2)

	net.eval()

	with torch.no_grad():
		for ii, large_sample_batched in enumerate(zip(*testloader_list, *testloader_flip_list)):
			print(ii)
			#1 0.5 0.75 1.25 1.5 1.75 ; flip:
			sample1 = large_sample_batched[:6]
			sample2 = large_sample_batched[6:]
			for iii,sample_batched in enumerate(zip(sample1,sample2)):

				inputs, labels_single = sample_batched[0]['image'], sample_batched[0]['label']
				inputs_f, labels_single_f = sample_batched[1]['image'], sample_batched[1]['label']
				inputs = torch.cat((inputs, inputs_f), dim=0)
				labels = torch.cat((labels_single, labels_single_f), dim=0)

				if iii == 0:
					_,_,h,w = inputs.size()
				# assert inputs.size() == inputs_f.size()

				# Forward pass of the mini-batch
				inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)

				with torch.no_grad():
					if gpu_id >= 0:
						inputs, labels, labels_single = inputs.cuda(), labels.cuda(), labels_single.cuda()
					# outputs = net.forward(inputs)
					# pdb.set_trace()
					outputs, outputs_aux = net.forward((inputs, num_dataset_lbl), training=False)

					# print(outputs.shape, outputs_aux.shape)
					if opts.dataset == 'cihp':
						outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
					elif opts.dataset == 'pascal':
						outputs = (outputs[0] + flip(outputs[1], dim=-1)) / 2
					else:
						outputs = (outputs[0] + flip(flip_atr(outputs[1]), dim=-1)) / 2

					outputs = outputs.unsqueeze(0)

					if iii>0:
						outputs = F.upsample(outputs,size=(h,w),mode='bilinear',align_corners=True)
						outputs_final = outputs_final + outputs
					else:
						outputs_final = outputs.clone()

			################ plot pic
			predictions = torch.max(outputs_final, 1)[1]
			prob_predictions = torch.max(outputs_final,1)[0]
			results = predictions.cpu().numpy()
			prob_results = prob_predictions.cpu().numpy()
			vis_res = decode_labels(results)

			parsing_im = Image.fromarray(vis_res[0])
			parsing_im.save(opts.output_path + vis_dir + '{}.png'.format(img_list[ii][:-1]))
			cv2.imwrite(opts.output_path + mat_dir + '{}.png'.format(img_list[ii][:-1]), results[0,:,:])

		# total_iou += utils.get_iou(predictions, labels)
	end_time = timeit.default_timer()
	print('time use for '+ str(ii) + ' is :' + str(end_time - start_time))

	# Eval
	pred_path = opts.output_path + mat_dir
	eval_with_numpy(pred_path=pred_path, gt_path=opts.gt_path,classes=opts.classes, txt_file=opts.txt_file, dataset=opts.dataset)


if __name__ == '__main__':
	opts = get_parser()
	main(opts)
	# pred_path = opts.output_path + '/atr_output/'
	# eval_with_numpy(pred_path=pred_path, gt_path=opts.gt_path,classes=opts.classes, txt_file=opts.txt_file, dataset=opts.dataset)
