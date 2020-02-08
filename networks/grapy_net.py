import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import graph_pyramid, deeplab_xception_synBN


class GrapyNet(
	deeplab_xception_synBN.DeepLabv3_plus):
	def __init__(self, nInputChannels=3, n_classes=7, os=16, hidden_layers=256):
		super(GrapyNet, self).__init__(nInputChannels=nInputChannels,
							 n_classes=n_classes,
							 os=os)

		self.hidden_layers = hidden_layers
		self.nclasses = n_classes

		self.transform = nn.Conv2d(hidden_layers, 256, kernel_size=1, stride=1)
		self.sem2 = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)

		self.gcn_network = graph_pyramid.GraphPyramidModule(cate_num1=2, cate_num2=5, cate_num3=n_classes)

	def set_category_list(self, list1, list2):
		self.gcn_network.set_cate_lis(list1, list2)

	def forward(self, input, training=True):

		x, low_level_features = self.xception_features(input)
		x1 = self.aspp1(x)
		x2 = self.aspp2(x)
		x3 = self.aspp3(x)
		x4 = self.aspp4(x)
		x5 = self.global_avg_pool(x)
		x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

		x = torch.cat((x1, x2, x3, x4, x5), dim=1)

		x = self.concat_projection_conv1(x)
		x = self.concat_projection_bn1(x)
		x = self.relu(x)

		x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

		low_level_features = self.feature_projection_conv1(low_level_features)
		low_level_features = self.feature_projection_bn1(low_level_features)
		low_level_features = self.relu(low_level_features)

		x = torch.cat((x, low_level_features), dim=1)

		features = self.decoder(x)

		x, x_aux = self.gcn_network(features=features, training=training)

		x = self.sem2(x)

		# this is for the final upsampling
		x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)
		x_aux = F.upsample(x_aux, size=input.size()[2:], mode='bilinear', align_corners=True)

		return x, x_aux

	def load_state_dict_new(self, state_dict, strict=True):
		r"""Copies parameters and buffers from :attr:`state_dict` into
		this module and its descendants. If :attr:`strict` is ``True``, then
		the keys of :attr:`state_dict` must exactly match the keys returned
		by this module's :meth:`~torch.nn.Module.state_dict` function.

		The purpose of the block is to copy some layers of different names from the pretrain model

		Arguments:
			state_dict (dict): a dict containing parameters and
				persistent buffers.
			strict (bool, optional): whether to strictly enforce that the keys
				in :attr:`state_dict` match the keys returned by this module's
				:meth:`~torch.nn.Module.state_dict` function. Default: ``True``
		"""
		missing_keys = []
		unexpected_keys = []
		error_msgs = []

		# copy state_dict so _load_from_state_dict can modify it
		metadata = getattr(state_dict, '_metadata', None)
		new_state_dict = state_dict.copy()

		# Here we break previous decoder into two parts
		for name, param in state_dict.items():
			if 'semantic' in name:
				name_1 = name.replace('semantic', 'gcn_network.sem')
				new_state_dict[name_1] = param

		if metadata is not None:
			new_state_dict._metadata = metadata

		def load(module, prefix=''):
			local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
			module._load_from_state_dict(
				new_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
			for name, child in module._modules.items():
				if child is not None:
					load(child, prefix + name + '.')

		load(self)

		if strict:
			error_msg = ''
			if len(unexpected_keys) > 0:
				error_msgs.insert(
					0, 'Unexpected key(s) in state_dict: {}. '.format(
						', '.join('"{}"'.format(k) for k in unexpected_keys)))
			if len(missing_keys) > 0:
				error_msgs.insert(
					0, 'Missing key(s) in state_dict: {}. '.format(
						', '.join('"{}"'.format(k) for k in missing_keys)))

		if len(error_msgs) > 0:
			raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
				self.__class__.__name__, "\n\t".join(error_msgs)))


class GrapyMutualLearning(
	deeplab_xception_synBN.DeepLabv3_plus_multi_set):
	def __init__(self, nInputChannels=3, os=16, hidden_layers=256):
		super(GrapyMutualLearning, self).__init__(nInputChannels=nInputChannels, os=os)

		self.hidden_layers = hidden_layers

		self.transform = nn.Conv2d(hidden_layers, 256, kernel_size=1, stride=1)
		self.gcn_network = graph_pyramid.GraphPyramidModuleML(cate_num1=2, cate_num2=5)

		self.sem2_cihp = nn.Conv2d(256, 20, kernel_size=1, stride=1)
		self.sem2_pascal = nn.Conv2d(256, 7, kernel_size=1, stride=1)
		self.sem2_atr = nn.Conv2d(256, 18, kernel_size=1, stride=1)

	def set_category_list(self, c1, c2, p1, p2, a1, a2):
		self.gcn_network.set_cate_lis(c1=c1, c2=c2, p1=p1, p2=p2, a1=a1, a2=a2)

	def forward(self, input, training=True):

		input, dataset = input

		if dataset == 0:
			sem2 = self.sem2_cihp

		elif dataset == 1:
			sem2 = self.sem2_pascal

		else:
			sem2 = self.sem2_atr

		x, low_level_features = self.xception_features(input)
		x1 = self.aspp1(x)
		x2 = self.aspp2(x)
		x3 = self.aspp3(x)
		x4 = self.aspp4(x)
		x5 = self.global_avg_pool(x)
		x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

		x = torch.cat((x1, x2, x3, x4, x5), dim=1)

		x = self.concat_projection_conv1(x)
		x = self.concat_projection_bn1(x)
		x = self.relu(x)

		x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

		low_level_features = self.feature_projection_conv1(low_level_features)
		low_level_features = self.feature_projection_bn1(low_level_features)
		low_level_features = self.relu(low_level_features)

		x = torch.cat((x, low_level_features), dim=1)

		features = self.decoder(x)
		x, x_aux = self.gcn_network(features=features, training=training, dataset=dataset)

		x = sem2(x)

		# this is for the final upsampling
		x_aux = F.upsample(x_aux, size=input.size()[2:], mode='bilinear', align_corners=True)
		x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

		return x, x_aux

	def load_state_dict_new(self, state_dict, strict=True):
		r"""Copies parameters and buffers from :attr:`state_dict` into
		this module and its descendants. If :attr:`strict` is ``True``, then
		the keys of :attr:`state_dict` must exactly match the keys returned
		by this module's :meth:`~torch.nn.Module.state_dict` function.

		The purpose of the block is to copy some layers of different names from the pretrain model

		Arguments:
			state_dict (dict): a dict containing parameters and
				persistent buffers.
			strict (bool, optional): whether to strictly enforce that the keys
				in :attr:`state_dict` match the keys returned by this module's
				:meth:`~torch.nn.Module.state_dict` function. Default: ``True``
		"""
		missing_keys = []
		unexpected_keys = []
		error_msgs = []

		# copy state_dict so _load_from_state_dict can modify it
		metadata = getattr(state_dict, '_metadata', None)
		new_state_dict = state_dict.copy()

		# Here we break previous decoder into two parts
		for name, param in state_dict.items():
			if 'semantic_aux_cihp' in name:
				name_1 = name.replace('semantic_aux_cihp', 'gcn_network.sem_cihp')
				new_state_dict[name_1] = param
			elif 'semantic_aux_pascal' in name:
				name_1 = name.replace('semantic_aux_pascal', 'gcn_network.sem_pascal')
				new_state_dict[name_1] = param
			elif 'semantic_aux_atr' in name:
				name_1 = name.replace('semantic_aux_atr', 'gcn_network.sem_atr')
				new_state_dict[name_1] = param

		if metadata is not None:
			new_state_dict._metadata = metadata

		def load(module, prefix=''):
			local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
			module._load_from_state_dict(
				new_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
			for name, child in module._modules.items():
				if child is not None:
					load(child, prefix + name + '.')

		load(self)

		if strict:
			error_msg = ''
			if len(unexpected_keys) > 0:
				error_msgs.insert(
					0, 'Unexpected key(s) in state_dict: {}. '.format(
						', '.join('"{}"'.format(k) for k in unexpected_keys)))
			if len(missing_keys) > 0:
				error_msgs.insert(
					0, 'Missing key(s) in state_dict: {}. '.format(
						', '.join('"{}"'.format(k) for k in missing_keys)))

		if len(error_msgs) > 0:
			raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
				self.__class__.__name__, "\n\t".join(error_msgs)))