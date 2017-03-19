# Import useful libraries
import cv2
import numpy as np

def crop_digits(bbox, size, extra_margin=0.15, recalibrate_bounding_boxes=True):

	num_digits = {}
	img_size = {}
	cropped_digits = {}
	input_data = {}

	for data_type in ['train', 'test']:
		# Find the number of digits in each image
		num_digits[data_type] = [len(bbox[data_type]['boxes'][i]) for i in range(size[data_type])]

		# Get the size of each image. The first value of each tuple is the height,
		# the second is the width and the third is the number of color channels
		# We will use only the first two elements of each tuple
		img_size[data_type] = [cv2.imread('SVHN/%s/%d.png' %(data_type,i)).shape for i in range(1, size[data_type]+1)]

		# Dictionary to store the cropped digits coordinates
		cropped_digits[data_type] = {}

		# Dictionary to store our cropped images
		input_data[data_type] = {}

		for i in range(size[data_type]):
			cropped_digits[data_type][i] = {}
			total_width = 0
			min_top = 99999
			max_top = -99999
			# The height is the same for all digits
			digit_height = bbox[data_type]['boxes'][i][0]['height']
			for k in range(num_digits[data_type][i]):
				w = bbox[data_type]['boxes'][i][k]['width']
				h = bbox[data_type]['boxes'][i][k]['height']
				t = bbox[data_type]['boxes'][i][k]['top']
				l = bbox[data_type]['boxes'][i][k]['left']
				total_width += w
				if k==0:
					extreme_left = max(0,l)
				if t<min_top:
					# Clip minimum top value at 0
					min_top = max(0,t)
				if t>max_top:
					# Clip maximum top value at the height of the image
					max_top = min(t, img_size[data_type][i][0])
			cropped_digits[data_type][i]['top'] = max(0,int(min_top - extra_margin*digit_height))
			cropped_digits[data_type][i]['left'] = max(0,int(extreme_left - extra_margin*total_width))
			cropped_digits[data_type][i]['width'] = int((1+2*extra_margin)*total_width)
			cropped_digits[data_type][i]['height'] = int(max_top + (1+2*extra_margin)*digit_height - min_top)
			# Read and save cropped image
			img = cv2.imread('SVHN/%s/%d.png' %(data_type,i+1))
			top = cropped_digits[data_type][i]['top']
			bottom = min(top + cropped_digits[data_type][i]['height'], img_size[data_type][i][0])
			left = cropped_digits[data_type][i]['left']
			right = min(left + cropped_digits[data_type][i]['width'], img_size[data_type][i][1])
			img = img[top:bottom,left:right]
			input_data[data_type][i] = img

	if recalibrate_bounding_boxes:
		# Recalibrate the bounding boxes of digits
		for data_type in ['train', 'test']:
			for i in range(size[data_type]):
				for k in range(num_digits[data_type][i]):
					bbox[data_type]['boxes'][i][k]['top'] -= cropped_digits[data_type][i]['top']
					bbox[data_type]['boxes'][i][k]['left'] -= cropped_digits[data_type][i]['left']

	return bbox, input_data

def get_regression_targets(input_data, size, bbox, left_null_digit=1.3, max_digits=5):
	reg_targets = {}
	img_size = {}
	num_digits = {}
	# Insert the normalized top-left and bottom-right coordinates
	# of each digit in the empty target dataset.
	for data_type in ['train','test']:
		# Initialize matrices for the targets of the training and testing datasets.
		reg_targets[data_type] = np.zeros((size[data_type], max_digits*4))
		# Find the number of digits in each image
		num_digits[data_type] = [len(bbox[data_type]['boxes'][i]) for i in range(size[data_type])]
		# Compute the size of the images (in case we have cropped the images)
		# First value is the height, second the width, third is the number of color channels
		img_size[data_type] = [input_data[data_type][i].shape for i in range(size[data_type])]
		for i in range(size[data_type]):
			m = 0
			n_digits_sample = num_digits[data_type][i]
			img_height = img_size[data_type][i][0]
			img_width = img_size[data_type][i][1]
			bbox_coord = bbox[data_type].boxes[i]
			for k in range(5):
				if (k+1)<=n_digits_sample:
					bounding_box = bbox_coord[k]
					top = bounding_box['top']/img_height
					left = bounding_box['left']/img_width
					bottom = (bounding_box['top']+bounding_box['height'])/img_height
					right = (bounding_box['left']+bounding_box['width'])/img_width
				else:
					bounding_box = bbox_coord[n_digits_sample-1]
					top = bounding_box['top']/img_height
					left = left_null_digit
					bottom = (bounding_box['top']+bounding_box['height'])/img_height
					right = left_null_digit+bounding_box['width']/img_width
				reg_targets[data_type][i,m:(m+4)] = left, top, right, bottom
				m += 4
	return reg_targets
	
def get_classification_targets(bbox, size, max_digits=5, null_target=10):
	# Empty dictionary to store classification targets
	cl_targets = {}
	# Empty dictionary to store the number of digits
	num_digits = {}
	for data_type in ['train','test']:
		# Compute the number of digits in each image
		num_digits[data_type] = [len(bbox[data_type]['boxes'][i]) for i in range(size[data_type])]
		# Initialize matrices with null_target
		cl_targets[data_type] = np.zeros((size[data_type],max_digits))
		cl_targets[data_type].fill(null_target)
		# Insert real target of non-null digits
		for i in range(size[data_type]):
			for k in range(num_digits[data_type][i]):
				if k<5:
					cl_targets[data_type][i,k] = bbox[data_type].boxes[i][k]['label']

	# One-hot encode the target labels in the training and testing set
	cl_ohe = {}
	for data_type in ['train','test']:
		cl_ohe[data_type] = {}
		for i in range(max_digits):
		    cl_ohe[data_type][i] = np.zeros((size[data_type], 11))
		    cl_ohe[data_type][i][np.arange(0,size[data_type]),cl_targets[data_type][:,i].astype(int)] = 1

	return cl_targets, cl_ohe

# Read the images in grayscale mode and rescale them
def gray_rescale(input_data, size, rescaled_width=64, rescaled_height=64):
	# Initialize input dataset
	X = {}
	img_size = {}
	for data_type in ['train','test']:
		X[data_type] = np.zeros((size[data_type], rescaled_height, rescaled_width))
		img_size[data_type] = [input_data[data_type][i].shape for i in range(size[data_type])]
		for i in range(size[data_type]):
			# Transform to grayscale. We simply take the mean value across the RGB channels
			img = input_data[data_type][i].mean(axis=2)
			# Check image size. We use INTER_AREA for interpolation if we're shrinking the image
			# Otherwise we use INTER_LINEAR for zooming
			if rescaled_height<img_size[data_type][i][0]:
				img = cv2.resize(img, (rescaled_height,rescaled_width), interpolation = cv2.INTER_AREA)
			else:
				img = cv2.resize(img,(rescaled_height, rescaled_width), interpolation = cv2.INTER_LINEAR)
			# Normalize data
			X[data_type][i,:,:] = img/255
		# Add a fourth dimension for the number of color channels.
		# Since we're working with grayscale images it will be just 1
		X[data_type] = np.reshape(X[data_type], (size[data_type], rescaled_height, rescaled_width, 1))
	return X

# Evaluate the performance on a dataset
def evaluate_performance(model, X, order_digits, cl_targets, size, data_type='train', max_digits=5):
	# Make predictions
	y_pred = model.predict(X[data_type])
	y_pred = np.hstack([np.argmax(y_pred[i], axis=1).reshape(-1,1) for i in range(max_digits)])

	# Compute the accuracy for each digit in part
	for i in range(max_digits):
		print('%s digit average accuracy on the entire %s dataset: '
		%(order_digits[i],data_type), np.round(
		np.mean(y_pred[:,i]==cl_targets[data_type][:,i])*100, 2), '%')

	# Compute the accuracy for the entire number
	num_correct_pred = (((y_pred==cl_targets[data_type]).sum(axis=1))==max_digits).sum()
	print('Average accuracy for the entire number on the %s dataset' %(data_type),
	np.round((num_correct_pred/size[data_type])*100, 2), '%\n')

# Function to compute IoU scores between predicted and actual values
def mean_iou(y_true, y_pred, k):
    # Coordinates overlapped area
    left = np.maximum(y_true[:,k],y_pred[:,k])
    top = np.maximum(y_true[:,k+1],y_pred[:,k+1])
    right = np.minimum(y_true[:,k+2],y_pred[:,k+2])
    bottom = np.minimum(y_true[:,k+3],y_pred[:,k+3])
    # Width and height overlapped area
    width = right - left
    height = bottom-top
    overlapped_area = width*height
    # If the width or height is negative the overlapped area is 0
    overlapped_area[(width<0)|(height<0)] = 0
    predicted_area = (y_pred[:,k+3]-y_pred[:,k+1])*(y_pred[:,k+2]-y_pred[:,k])
    true_area = (y_true[:,k+3]-y_true[:,k+1])*(y_true[:,k+2]-y_true[:,k])
    union_area = true_area+predicted_area-overlapped_area
    iou_scores = overlapped_area/(union_area)
    return iou_scores
