import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from networks import graph
import numpy as np
from sync_batchnorm import SynchronizedBatchNorm1d


class Self_Attn_instead_of_GCN_SYNBN_no_gamma(nn.Module):
	""" Self attention Layer"""

	def __init__(self, in_dim, activation):
		super(Self_Attn_instead_of_GCN_SYNBN_no_gamma, self).__init__()
		self.chanel_in = in_dim

		# But here we are not using activation function
		self.activation = activation

		self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		# self.gamma = nn.Parameter(torch.zeros(1))

		self.bn = SynchronizedBatchNorm1d(in_dim)

		self.softmax = nn.Softmax(dim=-1)  #

	def forward(self, x, adj=None):
		"""
			inputs :
				x : input feature maps( B X C X W X H)
			returns :
				out : self attention value + input feature
				attention: B X N X N (N is Width*Height)
		"""
		m_batchsize, C, class_num = x.size()
		proj_query = self.query_conv(x).view(m_batchsize, -1, class_num).permute(0, 2, 1)  # B X CX(N)
		proj_key = self.key_conv(x).view(m_batchsize, -1, class_num)  # B X C x (*W*H)
		energy = torch.bmm(proj_query, proj_key)  # transpose check
		attention = self.softmax(energy)  # BX (N) X (N)

		# Here we use synBN
		proj_value = self.value_conv(self.bn(x)).view(m_batchsize, -1, class_num)  # B X C X N
		# proj_value = self.value_conv(x).view(m_batchsize, -1, class_num)  # B X C X N
		out = torch.bmm(proj_value, attention.permute(0, 2, 1))

		out = out.view(m_batchsize, C, class_num)
		out = out + x

		return out, attention


class GraphContextReasoning(nn.Module):
	""" Self attention Layer"""

	def __init__(self, in_dim, activation):
		super(GraphContextReasoning, self).__init__()
		self.chanel_in = in_dim

		# But here we are not using activation function
		self.activation = activation

		self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		# self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax = nn.Softmax(dim=-1)  #

	def forward(self, x, adj=None):
		"""
			inputs :
				x : input feature maps( B X C X W X H)
			returns :
				out : self attention value + input feature
				attention: B X N X N (N is Width*Height)
		"""
		m_batchsize, C, class_num = x.size()
		proj_query = self.query_conv(x).view(m_batchsize, -1, class_num).permute(0, 2, 1)  # B X CX(N)
		proj_key = self.key_conv(x).view(m_batchsize, -1, class_num)  # B X C x (*W*H)
		energy = torch.bmm(proj_query, proj_key)  # transpose check
		attention = self.softmax(energy)  # BX (N) X (N)

		# Here we use synBN
		proj_value = self.value_conv(x).view(m_batchsize, -1, class_num)  # B X C X N
		# proj_value = self.value_conv(x).view(m_batchsize, -1, class_num)  # B X C X N
		out = torch.bmm(proj_value, attention.permute(0, 2, 1))

		out = out.view(m_batchsize, C, class_num)
		out = out + x

		return out, attention


if __name__ == '__main__':
	graph = torch.randn((7, 128))
	pred = (torch.rand((7, 7)) * 7).int()
	# a = en.forward(graph,pred)
	# print(a.size())
