import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import gcn

from sync_batchnorm import SynchronizedBatchNorm1d


# Blocks
class LevelReasoning(nn.Module):
	def __init__(self, n_classes=7, hidden_layers=256, graph_hidden_layers=512):
		super(LevelReasoning, self).__init__()

		self.hidden_layers = hidden_layers
		self.nclasses = n_classes

		self.graph_transfer = nn.Conv1d(graph_hidden_layers, graph_hidden_layers, kernel_size=1, stride=1)
		self.graph_transfer_back = nn.Conv1d(graph_hidden_layers, hidden_layers, kernel_size=1, stride=1)

		self.attn1 = gcn.GraphContextReasoning(graph_hidden_layers, 'relu')
		self.attn2 = gcn.GraphContextReasoning(graph_hidden_layers, 'relu')
		self.attn3 = gcn.GraphContextReasoning(graph_hidden_layers, 'relu')

		self.bn1 = SynchronizedBatchNorm1d(graph_hidden_layers)
		self.bn2 = SynchronizedBatchNorm1d(graph_hidden_layers)
		self.bn3 = SynchronizedBatchNorm1d(graph_hidden_layers)

	def forward(self, x, training=True):

		x = self.graph_transfer(x.permute(0, 2, 1))

		x, att_map1 = self.attn1(self.bn1(x))

		x = F.relu(x)

		x, att_map2 = self.attn2(self.bn2(x))

		x = F.relu(x)

		x, att_map3 = self.attn3(self.bn3(x)) # batch * feature channels * classnum

		x = F.relu(x)

		x = self.graph_transfer_back(x)

		return x


class GraphPyramidModule(nn.Module):
	def __init__(self, hidden_layer1=256, hidden_layer2=256, hidden_layer3=256, graph_hidden_layers=512,
				 cate_num1=2, cate_num2=5, cate_num3=20):
		super(GraphPyramidModule, self).__init__()

		# Hard coded feature numbers
		self.gcn_block_feature1 = LevelReasoning(cate_num1, hidden_layers=hidden_layer1, graph_hidden_layers=graph_hidden_layers)
		self.gcn_block_feature2 = LevelReasoning(cate_num2, hidden_layers=hidden_layer2, graph_hidden_layers=graph_hidden_layers)
		self.gcn_block_feature3 = LevelReasoning(cate_num3, hidden_layers=hidden_layer3, graph_hidden_layers=graph_hidden_layers)

		self.cate_num3 = cate_num3
		self.cate_num2 = cate_num2
		self.cate_num1 = cate_num1

		self.hidden_layer1 = hidden_layer1
		self.hidden_layer2 = hidden_layer2
		self.hidden_layer3 = hidden_layer3

		self.graph_hidden_layers = graph_hidden_layers

		self.fusion = nn.Sequential(nn.Conv2d(256 * 4, 256 * 2, kernel_size=1, stride=1),
									nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1),
									nn.ReLU())

		self.transform_lv1 = nn.Conv2d(hidden_layer1, 256, kernel_size=1, stride=1)
		self.transform_lv2 = nn.Conv2d(hidden_layer2, 256, kernel_size=1, stride=1)
		self.transform_lv3 = nn.Conv2d(hidden_layer3, 256, kernel_size=1, stride=1)

		self.pooling_softmax = torch.nn.Softmax(dim=1)

		self.sem = nn.Conv2d(256, cate_num3, kernel_size=1, stride=1)

		self.cate_list1 = []
		self.cate_list2 = []

	def set_cate_lis(self, list1, list2):
		self.cate_list1 = list1
		self.cate_list2 = list2

	def mask2map(self, mask, class_num):
		# print(featuremap.shape, mask.shape)
		# print(mask.shape)
		n, h, w = mask.shape

		maskmap_ave = torch.zeros(n, class_num, h, w).cuda()
		maskmap_max = torch.zeros(n, class_num, h, w).cuda()

		# print(maskmap, maskmap.shape)

		for i in range(class_num):
			# print(i)
			# print(mask)
			class_pix = torch.where(mask == i, torch.ones(1).cuda(), torch.zeros(1).cuda())
			# print(temp)

			class_sum = torch.sum(class_pix.view(n, h * w), dim=1)
			# print(temp_sum)
			class_sum = torch.where(class_sum == 0, torch.ones(1).cuda(), class_sum)

			# print(map.shape, sum.shape)
			class_pix_ave = class_pix / class_sum.view(n, 1, 1)

			# print(temp)

			maskmap_ave[:, i, :, :] = class_pix_ave
			maskmap_max[:, i, :, :] = class_pix

		return maskmap_ave, maskmap_max

	def mask2catemask(self, mask, cate_list):

		cate_num = len(cate_list)

		for j in range(cate_num):
			for i in cate_list[j]:
				# print(mask.type(), torch.tensor([j], dtype=torch.int32).type())
				mask = torch.where(mask == i, torch.tensor([j]).cuda(), mask)

		return mask

	def graph_semantic_aggregation(self, mask, features, cate_list=None, label_class=20):

		if cate_list is not None:
			class_num = len(cate_list)
			raw_mask = self.mask2catemask(mask, cate_list)

		else:
			class_num = label_class
			raw_mask = mask

		maskmap_ave, maskmap_heatmap = self.mask2map(raw_mask, class_num)
		n_batch, c_channel, h_input, w_input = features.size()

		max_lis = []
		for i in range(class_num):
			class_max = features * maskmap_heatmap[:, i, :, :].unsqueeze(1)

			class_max = torch.max(class_max.view(n_batch, c_channel, h_input * w_input), -1)[0]
			max_lis.append(class_max.unsqueeze(1))

		features_max = torch.cat(max_lis, 1)
		features_ave = torch.matmul(maskmap_ave.view(n_batch, class_num, h_input * w_input),
									# batch * class_num * hw
									features.permute(0, 2, 3, 1).view(n_batch, h_input * w_input, 256)
									# batch * hw * feature channels
									)  # batch * classnum * feature channels

		return torch.cat([features_ave, features_max], 2), maskmap_heatmap

	def forward(self, features=None, training=True):

		n, c, h, w = features.size()
		x_aux = self.sem(features)
		raw_mask = torch.argmax(x_aux, 1)

		(graph_features_lv1, mask_lv1) = self.graph_semantic_aggregation(raw_mask, features, cate_list=self.cate_list1, label_class=self.cate_num1)

		graph_features_lv1 = self.gcn_block_feature1(graph_features_lv1, training=training)
		features_lv1 = torch.matmul(graph_features_lv1, mask_lv1.view(n, self.cate_num1, h * w)).view(n, self.hidden_layer1, h, w) # batch * feature channels * h * w
		fused_feature1 = features + self.transform_lv1(features_lv1)

		(graph_features_lv2, mask_lv2) = self.graph_semantic_aggregation(raw_mask, fused_feature1, cate_list=self.cate_list2, label_class=self.cate_num2)
		graph_features_lv2 = self.gcn_block_feature2(graph_features_lv2, training=training)
		features_lv2 = torch.matmul(graph_features_lv2, mask_lv2.view(n, self.cate_num2, h * w)).view(n, self.hidden_layer2, h, w) # batch * feature channels * h * w
		fused_feature2 = fused_feature1 + self.transform_lv2(features_lv2)

		(graph_features_lv3, mask_lv3) = self.graph_semantic_aggregation(raw_mask, fused_feature2, label_class=self.cate_num3)
		graph_features_lv3 = self.gcn_block_feature3(graph_features_lv3, training=training)
		features_lv3 = torch.matmul(graph_features_lv3, mask_lv3.view(n, self.cate_num3, h * w)).view(n, self.hidden_layer3, h, w) # batch * feature channels * h * w
		fused_feature3 = fused_feature2 + self.transform_lv3(features_lv3)

		fused_feature = torch.cat([features, fused_feature1, fused_feature2, fused_feature3], 1)

		features = self.fusion(fused_feature)

		return features, x_aux


class GraphPyramidModuleML_res(nn.Module):
	def __init__(self, hidden_layer1=256, hidden_layer2=256, hidden_layer3=256, graph_hidden_layers=512,
				 cate_num1=2, cate_num2=5):
		super(GraphPyramidModuleML_res, self).__init__()

		# Hard coded feature numbers
		self.gcn_block_feature1 = LevelReasoning(cate_num1, hidden_layers=hidden_layer1, graph_hidden_layers=graph_hidden_layers)
		self.gcn_block_feature2 = LevelReasoning(cate_num2, hidden_layers=hidden_layer2, graph_hidden_layers=graph_hidden_layers)

		self.gcn_block_feature_cihp = LevelReasoning(20, hidden_layers=hidden_layer3, graph_hidden_layers=graph_hidden_layers)
		self.gcn_block_feature_pascal = LevelReasoning(7, hidden_layers=hidden_layer3, graph_hidden_layers=graph_hidden_layers)
		self.gcn_block_feature_atr = LevelReasoning(18, hidden_layers=hidden_layer3, graph_hidden_layers=graph_hidden_layers)

		self.cate_num2 = cate_num2
		self.cate_num1 = cate_num1

		self.hidden_layer1 = hidden_layer1
		self.hidden_layer2 = hidden_layer2
		self.hidden_layer3 = hidden_layer3

		self.graph_hidden_layers = graph_hidden_layers

		self.fusion = nn.Sequential(nn.Conv2d(256 * 4, 256 * 2, kernel_size=1, stride=1),
									nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1),
									nn.ReLU())

		self.transform_lv1 = nn.Conv2d(hidden_layer1, 256, kernel_size=1, stride=1)
		self.transform_lv2 = nn.Conv2d(hidden_layer2, 256, kernel_size=1, stride=1)
		self.transform_lv3 = nn.Conv2d(hidden_layer3, 256, kernel_size=1, stride=1)

		self.pooling_softmax = torch.nn.Softmax(dim=1)

		self.sem_cihp = nn.Conv2d(256, 20, kernel_size=1, stride=1)
		self.sem_pascal = nn.Conv2d(256, 7, kernel_size=1, stride=1)
		self.sem_atr = nn.Conv2d(256, 18, kernel_size=1, stride=1)

		self.cate_list_cihp1 = []
		self.cate_list_cihp2 = []

		self.cate_list_pascal1 = []
		self.cate_list_pascal2 = []

		self.cate_list_atr1 = []
		self.cate_list_atr2 = []

	def set_cate_lis(self, c1, c2, p1, p2, a1, a2):

		self.cate_list_cihp1 = c1
		self.cate_list_cihp2 = c2

		self.cate_list_pascal1 = p1
		self.cate_list_pascal2 = p2

		self.cate_list_atr1 = a1
		self.cate_list_atr2 = a2

	def mask2map(self, mask, class_num):
		# print(featuremap.shape, mask.shape)
		# print(mask.shape)
		n, h, w = mask.shape

		maskmap_ave = torch.zeros(n, class_num, h, w).cuda()
		maskmap_max = torch.zeros(n, class_num, h, w).cuda()

		# print(maskmap, maskmap.shape)

		for i in range(class_num):
			# print(i)
			# print(mask)
			class_pix = torch.where(mask == i, torch.ones(1).cuda(), torch.zeros(1).cuda())
			# print(temp)

			class_sum = torch.sum(class_pix.view(n, h * w), dim=1)
			# print(temp_sum)
			class_sum = torch.where(class_sum == 0, torch.ones(1).cuda(), class_sum)

			# print(map.shape, sum.shape)
			class_pix_ave = class_pix / class_sum.view(n, 1, 1)

			# print(temp)

			maskmap_ave[:, i, :, :] = class_pix_ave
			maskmap_max[:, i, :, :] = class_pix

		return maskmap_ave, maskmap_max

	def mask2catemask(self, mask, cate_list):

		cate_num = len(cate_list)

		for j in range(cate_num):
			for i in cate_list[j]:
				# print(mask.type(), torch.tensor([j], dtype=torch.int32).type())
				mask = torch.where(mask == i, torch.tensor([j]).cuda(), mask)

		return mask

	def graph_semantic_aggregation(self, mask, features, cate_list=None, label_class=20):

		if cate_list is not None:
			class_num = len(cate_list)
			raw_mask = self.mask2catemask(mask, cate_list)

		else:
			class_num = label_class
			raw_mask = mask

		maskmap_ave, maskmap_heatmap = self.mask2map(raw_mask, class_num)
		n_batch, c_channel, h_input, w_input = features.size()

		max_lis = []
		for i in range(class_num):
			class_max = features * maskmap_heatmap[:, i, :, :].unsqueeze(1)

			class_max = torch.max(class_max.view(n_batch, c_channel, h_input * w_input), -1)[0]
			max_lis.append(class_max.unsqueeze(1))

		features_max = torch.cat(max_lis, 1)
		features_ave = torch.matmul(maskmap_ave.view(n_batch, class_num, h_input * w_input),
									# batch * class_num * hw
									features.permute(0, 2, 3, 1).view(n_batch, h_input * w_input, 256)
									# batch * hw * feature channels
									)  # batch * classnum * feature channels

		return torch.cat([features_ave, features_max], 2), maskmap_heatmap

	def forward(self, features=None, dataset=0, training=True):

		if dataset == 0:
			sem = self.sem_cihp
			label_class = 20
			gcn_block_feature_specific = self.gcn_block_feature_cihp

			cate_list1 = self.cate_list_cihp1
			cate_list2 = self.cate_list_cihp2

		elif dataset == 1:
			sem = self.sem_pascal
			label_class = 7
			gcn_block_feature_specific = self.gcn_block_feature_pascal

			cate_list1 = self.cate_list_pascal1
			cate_list2 = self.cate_list_pascal2

		else:
			sem = self.sem_atr
			label_class = 18
			gcn_block_feature_specific = self.gcn_block_feature_atr

			cate_list1 = self.cate_list_atr1
			cate_list2 = self.cate_list_atr2

		n, c, h, w = features.size()

		x_aux = sem(features)

		raw_mask = torch.argmax(x_aux, 1)

		(graph_features_lv1, mask_lv1) = self.graph_semantic_aggregation(raw_mask, features, cate_list=cate_list1)

		graph_features_lv1 = self.gcn_block_feature1(graph_features_lv1, training=training)
		features_lv1 = torch.matmul(graph_features_lv1, mask_lv1.view(n, self.cate_num1, h * w)).view(n, self.hidden_layer1, h, w) # batch * feature channels * h * w
		fused_feature1 = features + self.transform_lv1(features_lv1)

		(graph_features_lv2, mask_lv2) = self.graph_semantic_aggregation(raw_mask, fused_feature1, cate_list=cate_list2)
		graph_features_lv2 = self.gcn_block_feature2(graph_features_lv2, training=training)
		features_lv2 = torch.matmul(graph_features_lv2, mask_lv2.view(n, self.cate_num2, h * w)).view(n, self.hidden_layer2, h, w) # batch * feature channels * h * w
		fused_feature2 = fused_feature1 + self.transform_lv2(features_lv2)

		(graph_features_lv3, mask_lv3) = self.graph_semantic_aggregation(raw_mask, fused_feature2, label_class=label_class)
		graph_features_lv3 = gcn_block_feature_specific(graph_features_lv3, training=training)
		features_lv3 = torch.matmul(graph_features_lv3, mask_lv3.view(n, label_class, h * w)).view(n, self.hidden_layer3, h, w) # batch * feature channels * h * w
		fused_feature3 = fused_feature2 + self.transform_lv3(features_lv3)

		fused_feature = torch.cat([features, fused_feature1, fused_feature2, fused_feature3], 1)

		features = features + self.fusion(fused_feature)

		return features, x_aux


class GraphPyramidModuleML(nn.Module):
	def __init__(self, hidden_layer1=256, hidden_layer2=256, hidden_layer3=256, graph_hidden_layers=512,
				 cate_num1=2, cate_num2=5):
		super(GraphPyramidModuleML, self).__init__()

		# Hard coded feature numbers
		self.gcn_block_feature1 = LevelReasoning(cate_num1, hidden_layers=hidden_layer1, graph_hidden_layers=graph_hidden_layers)
		self.gcn_block_feature2 = LevelReasoning(cate_num2, hidden_layers=hidden_layer2, graph_hidden_layers=graph_hidden_layers)

		self.gcn_block_feature_cihp = LevelReasoning(20, hidden_layers=hidden_layer3, graph_hidden_layers=graph_hidden_layers)
		self.gcn_block_feature_pascal = LevelReasoning(7, hidden_layers=hidden_layer3, graph_hidden_layers=graph_hidden_layers)
		self.gcn_block_feature_atr = LevelReasoning(18, hidden_layers=hidden_layer3, graph_hidden_layers=graph_hidden_layers)

		self.cate_num2 = cate_num2
		self.cate_num1 = cate_num1

		self.hidden_layer1 = hidden_layer1
		self.hidden_layer2 = hidden_layer2
		self.hidden_layer3 = hidden_layer3

		self.graph_hidden_layers = graph_hidden_layers

		self.fusion = nn.Sequential(nn.Conv2d(256 * 4, 256 * 2, kernel_size=1, stride=1),
									nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1),
									nn.ReLU())

		self.transform_lv1 = nn.Conv2d(hidden_layer1, 256, kernel_size=1, stride=1)
		self.transform_lv2 = nn.Conv2d(hidden_layer2, 256, kernel_size=1, stride=1)
		self.transform_lv3 = nn.Conv2d(hidden_layer3, 256, kernel_size=1, stride=1)

		self.pooling_softmax = torch.nn.Softmax(dim=1)

		self.sem_cihp = nn.Conv2d(256, 20, kernel_size=1, stride=1)
		self.sem_pascal = nn.Conv2d(256, 7, kernel_size=1, stride=1)
		self.sem_atr = nn.Conv2d(256, 18, kernel_size=1, stride=1)

		self.cate_list_cihp1 = []
		self.cate_list_cihp2 = []

		self.cate_list_pascal1 = []
		self.cate_list_pascal2 = []

		self.cate_list_atr1 = []
		self.cate_list_atr2 = []

	def set_cate_lis(self, c1, c2, p1, p2, a1, a2):

		self.cate_list_cihp1 = c1
		self.cate_list_cihp2 = c2

		self.cate_list_pascal1 = p1
		self.cate_list_pascal2 = p2

		self.cate_list_atr1 = a1
		self.cate_list_atr2 = a2

	def mask2map(self, mask, class_num):
		# print(featuremap.shape, mask.shape)
		# print(mask.shape)
		n, h, w = mask.shape

		maskmap_ave = torch.zeros(n, class_num, h, w).cuda()
		maskmap_max = torch.zeros(n, class_num, h, w).cuda()

		# print(maskmap, maskmap.shape)

		for i in range(class_num):
			# print(i)
			# print(mask)
			class_pix = torch.where(mask == i, torch.ones(1).cuda(), torch.zeros(1).cuda())
			# print(temp)

			class_sum = torch.sum(class_pix.view(n, h * w), dim=1)
			# print(temp_sum)
			class_sum = torch.where(class_sum == 0, torch.ones(1).cuda(), class_sum)

			# print(map.shape, sum.shape)
			class_pix_ave = class_pix / class_sum.view(n, 1, 1)

			# print(temp)

			maskmap_ave[:, i, :, :] = class_pix_ave
			maskmap_max[:, i, :, :] = class_pix

		return maskmap_ave, maskmap_max

	def mask2catemask(self, mask, cate_list):

		cate_num = len(cate_list)

		for j in range(cate_num):
			for i in cate_list[j]:
				# print(mask.type(), torch.tensor([j], dtype=torch.int32).type())
				mask = torch.where(mask == i, torch.tensor([j]).cuda(), mask)

		return mask

	def graph_semantic_aggregation(self, mask, features, cate_list=None, label_class=20):

		if cate_list is not None:
			class_num = len(cate_list)
			raw_mask = self.mask2catemask(mask, cate_list)

		else:
			class_num = label_class
			raw_mask = mask

		maskmap_ave, maskmap_heatmap = self.mask2map(raw_mask, class_num)
		n_batch, c_channel, h_input, w_input = features.size()

		max_lis = []
		for i in range(class_num):
			class_max = features * maskmap_heatmap[:, i, :, :].unsqueeze(1)

			class_max = torch.max(class_max.view(n_batch, c_channel, h_input * w_input), -1)[0]
			max_lis.append(class_max.unsqueeze(1))

		features_max = torch.cat(max_lis, 1)
		features_ave = torch.matmul(maskmap_ave.view(n_batch, class_num, h_input * w_input),
									# batch * class_num * hw
									features.permute(0, 2, 3, 1).view(n_batch, h_input * w_input, 256)
									# batch * hw * feature channels
									)  # batch * classnum * feature channels

		return torch.cat([features_ave, features_max], 2), maskmap_heatmap

	def forward(self, features=None, dataset=0, training=True):

		if dataset == 0:
			sem = self.sem_cihp
			label_class = 20
			gcn_block_feature_specific = self.gcn_block_feature_cihp

			cate_list1 = self.cate_list_cihp1
			cate_list2 = self.cate_list_cihp2

		elif dataset == 1:
			sem = self.sem_pascal
			label_class = 7
			gcn_block_feature_specific = self.gcn_block_feature_pascal

			cate_list1 = self.cate_list_pascal1
			cate_list2 = self.cate_list_pascal2

		else:
			sem = self.sem_atr
			label_class = 18
			gcn_block_feature_specific = self.gcn_block_feature_atr

			cate_list1 = self.cate_list_atr1
			cate_list2 = self.cate_list_atr2

		n, c, h, w = features.size()

		x_aux = sem(features)

		raw_mask = torch.argmax(x_aux, 1)

		(graph_features_lv1, mask_lv1) = self.graph_semantic_aggregation(raw_mask, features, cate_list=cate_list1)

		graph_features_lv1 = self.gcn_block_feature1(graph_features_lv1, training=training)
		features_lv1 = torch.matmul(graph_features_lv1, mask_lv1.view(n, self.cate_num1, h * w)).view(n, self.hidden_layer1, h, w) # batch * feature channels * h * w
		fused_feature1 = features + self.transform_lv1(features_lv1)

		(graph_features_lv2, mask_lv2) = self.graph_semantic_aggregation(raw_mask, fused_feature1, cate_list=cate_list2)
		graph_features_lv2 = self.gcn_block_feature2(graph_features_lv2, training=training)
		features_lv2 = torch.matmul(graph_features_lv2, mask_lv2.view(n, self.cate_num2, h * w)).view(n, self.hidden_layer2, h, w) # batch * feature channels * h * w
		fused_feature2 = fused_feature1 + self.transform_lv2(features_lv2)

		(graph_features_lv3, mask_lv3) = self.graph_semantic_aggregation(raw_mask, fused_feature2, label_class=label_class)
		graph_features_lv3 = gcn_block_feature_specific(graph_features_lv3, training=training)
		features_lv3 = torch.matmul(graph_features_lv3, mask_lv3.view(n, label_class, h * w)).view(n, self.hidden_layer3, h, w) # batch * feature channels * h * w
		fused_feature3 = fused_feature2 + self.transform_lv3(features_lv3)

		fused_feature = torch.cat([features, fused_feature1, fused_feature2, fused_feature3], 1)

		features = self.fusion(fused_feature)

		return features, x_aux
