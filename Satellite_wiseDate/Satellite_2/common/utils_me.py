# call for new time sseries

#train_inputs = TimeSeriesTensor(train, {'Y':(range(-HORIZON+1, 1), ['load', 'temp'])}, {'X':(range(-T+1, 1), ['load', 'temp'])})




import numpy as np
import pandas as pd
import os
from collections import UserDict

def load_data(data_dir):
	"""Load the GEFCom 2014 energy load data"""

	energy = pd.read_csv(os.path.join(data_dir, 'energy.csv'), parse_dates=['timestamp'])

	# Reindex the dataframe such that the dataframe has a record for every time point
	# between the minimum and maximum timestamp in the time series. This helps to 
	# identify missing time periods in the data (there are none in this dataset).

	energy.index = energy['timestamp']
	energy = energy.reindex(pd.date_range(min(energy['timestamp']),max(energy['timestamp']),freq='H'))
	energy = energy.drop('timestamp', axis=1)

	return energy


def mape(predictions, actuals):
	"""Mean absolute percentage error"""
	return ((predictions - actuals).abs() / actuals).mean()


def create_evaluation_df(predictions, test_inputs, H, scaler):
	"""Create a data frame for easy evaluation"""
	print("printing scaler shape " + str(len(scaler)))
	list_names = ['t+'+str(t) for t in range(1, H+1)]
	final_names = []
	for i in list_names:
		final_names.append(i + '_sqrt_A')
		final_names.append(i + '_e')
		final_names.append(i + '_M0')
		final_names.append(i + '_i0')
		final_names.append(i + '_omega')
		final_names.append(i + '_OMEGA')		
		
	print(final_names)
	"""list_names1 = ['t+1_sqrt_A', 't+1_e', 't+1_M0' ,' t+1_i0' , 't+1_omega' , 't+1_OMEGA' ,'t+2_sqrt_A', 't+2_e', 't+2_M0' ,' t+2_i0' , 't+2_omega' , 't+2_OMEGA' , 't+3_sqrt_A', 't+3_e','t+3_M0' ,' t+3_i0' , 't+3_omega' , 't+3_OMEGA', 't+4_sqrt_A', 't+4_e','t+4_M0' ,' t+4_i0' , 't+4_omega' , 't+4_OMEGA' ,'t+5_sqrt_A', 't+5_e' ,'t+5_M0' ,' t+5_i0' , 't+5_omega' , 't+5_OMEGA' ]"""
	eval_df = pd.DataFrame(predictions, columns=final_names)
	print(eval_df.head())
	print("********")		
	eval_df['Epoch_Time_of_Clock'] = test_inputs.dataframe.index
	print(eval_df.shape)	
	print(eval_df['Epoch_Time_of_Clock'])
	print(eval_df.head())
	eval_df.to_csv('vindeep1.csv')
	eval_df = pd.melt(eval_df, id_vars=['Epoch_Time_of_Clock'], value_name='prediction', var_name='h' , value_vars = final_names )
	eval_df.to_csv('vindeep11.csv')
	print("printing df.head after melting")
	print(eval_df.head())	
	a = eval_df['prediction']
	a = list(a)
	"""
	for index,key in enumerate(a):
		d = []
		d.append(key)
		b = np.array(d )

		a = y_scaler1.inverse_transform(b)

		if (index%2) == 0:
			final_a.append(scaler[0].inverse_tranform(b)[0])
		else:
			final_a.append(scaler[1].inverse_tranform(b))

	"""
	list_me = ['sqrt_A' , 'e' , 'M0' , 'i0' , 'omega' , 'OMEGA' ]
	
	for i in range(eval_df.shape[0]):
		k = -1
		for index , key in enumerate(list_me):
			if key in eval_df.iloc[i , 1]:
				k = index
				break
		d = []
		d.append(eval_df.iloc[i , 2])
		b = np.array(d)
		eval_df.iloc[i , 2] = scaler[k].inverse_transform(b)[0]	
	

					
			

	eval_df['actual'] = np.transpose(test_inputs['Y']).ravel()
	b = eval_df['actual']
	b = list(b)
	final_b = []
	"""
	for index,key in enumerate(b):
		if (index%2) == 0:
			final_b.append(scaler[0].inverse_tranform(key))
		else:
			final_b.append(scaler[1].inverse_tranform(key))
	"""

	for i in range(eval_df.shape[0]):
		k = -1
		for index , key in enumerate(list_me):
			if key in eval_df.iloc[i , 1]:
				k = index
				break
		d = []
		d.append(eval_df.iloc[i , 3])
		b = np.array(d)
		eval_df.iloc[i , 3] = scaler[k].inverse_transform(b)[0]	
	
	#eval_df[['prediction', 'actual']] = [final_a , final_b]
	
	#eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
	return eval_df


class TimeSeriesTensor(UserDict):
	"""A dictionary of tensors for input into the RNN model.
	
	Use this class to:
	  1. Shift the values of the time series to create a Pandas dataframe containing all the data
		 for a single training example
	  2. Discard any samples with missing values
	  3. Transform this Pandas dataframe into a numpy array of shape 
		 (samples, time steps, features) for input into Keras
	The class takes the following parameters:
	   - **dataset**: original time series
	   - **target** name of the target column
	   - **H**: the forecast horizon
	   - **tensor_structures**: a dictionary discribing the tensor structure of the form
			 { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }
			 if features are non-sequential and should not be shifted, use the form
			 { 'tensor_name' : (None, [feature, feature, ...])}
	   - **freq**: time series frequency (default 'H' - hourly)
	   - **drop_incomplete**: (Boolean) whether to drop incomplete samples (default True)
	"""
	
	def __init__(self, dataset, target_structure, tensor_structure, freq='H'):
		self.dataset = dataset
		self.target_structure = target_structure
		self.tensor_structure = tensor_structure
		self.target_names = list(target_structure.keys())
		self.tensor_names = list(tensor_structure.keys())
		self.dataframe = self._shift_data( freq)
		self.data = self._df2tensors(self.dataframe)
	
	def _shift_data(self, freq):		
		
		# Use the tensor_structures definitions to shift the features in the original dataset.
		# The result is a Pandas dataframe with multi-index columns in the hierarchy
		#     tensor - the name of the input tensor
		#     feature - the input feature to be shifted
		#     time step - the time step for the RNN in which the data is input. These labels
		#         are centred on time t. the forecast creation time
		
		drop_incomplete = True				
		df = self.dataset.copy()
		idx_tuples = []
		# for y
		for name, structure in self.target_structure.items():
			rng = structure[0]
			dataset_cols = structure[1]
			for col in dataset_cols:
			# do not shift non-sequential 'static' features
				if rng is None:
					df['context_'+col] = df[col]
					idx_tuples.append((name, col, 'static'))
				else:
					for t in rng:
						sign = '+' if t > 0 else ''
						shift = str(t) if t != 0 else ''
						period = 't'+sign+shift
						shifted_col = name+'_'+col+'_'+period
						df[shifted_col] = df[col].shift(t*-1, freq=freq)
						idx_tuples.append((name, col, period))

# for  x 
		for name, structure in self.tensor_structure.items():
			rng = structure[0]
			dataset_cols = structure[1]
			for col in dataset_cols:
			# do not shift non-sequential 'static' features
				if rng is None:
					df['context_'+col] = df[col]
					idx_tuples.append((name, col, 'static'))
				else:
					for t in rng:
						sign = '+' if t > 0 else ''
						shift = str(t) if t != 0 else ''
						period = 't'+sign+shift
						shifted_col = name+'_'+col+'_'+period
						df[shifted_col] = df[col].shift(t*-1, freq=freq)
						idx_tuples.append((name, col, period))
		df = df.drop(self.dataset.columns, axis=1)
		idx = pd.MultiIndex.from_tuples(idx_tuples, names=['tensor', 'feature', 'time step'])
		df.columns = idx
		if drop_incomplete:
			df = df.dropna(how='any')
		return df

	
	def _df2tensors(self, dataframe):
		# Transform the shifted Pandas dataframe into the multidimensional numpy arrays. These
		# arrays can be used to input into the keras model and can be accessed by tensor name.
		# For example, for a TimeSeriesTensor object named "model_inputs" and a tensor named
		# "target", the input tensor can be acccessed with model_inputs['target']
		inputs = {}
		#for y
		for name, structure in self.target_structure.items():
			rng = structure[0]
			cols = structure[1]
			tensor = dataframe[name][cols].as_matrix()
			if rng is None:
				tensor = tensor.reshape(tensor.shape[0], len(cols))
			else:
				tensor = tensor.reshape(tensor.shape[0], len(cols),len(rng))
				tensor = np.transpose(tensor, axes=[0, 2, 1])
				tensor = tensor.reshape(tensor.shape[0] , -1)
			inputs[name] = tensor

		
		#for  x
		for name, structure in self.tensor_structure.items():
			rng = structure[0]
			cols = structure[1]
			tensor = dataframe[name][cols].as_matrix()
			if rng is None:
				tensor = tensor.reshape(tensor.shape[0], len(cols))
			else:
				tensor = tensor.reshape(tensor.shape[0], len(cols), len(rng))
				tensor = np.transpose(tensor, axes=[0, 2, 1])
			inputs[name] = tensor
		return inputs
	   
	def subset_data(self, new_dataframe):
		
		# Use this function to recreate the input tensors if the shifted dataframe
		# has been filtered.
		
		self.dataframe = new_dataframe
		self.data = self._df2tensors(self.dataframe)
