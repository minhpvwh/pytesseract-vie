import os
import cv2
import glob
import tqdm
import argparse
from skimage.filters import threshold_local
import pytesseract
import numpy as np
import random

def check_exist(path):
	try:
		if not os.path.exists(path):
			os.mkdir(path)
	except Exception:
		raise ("please check your folder again")
		pass

def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final


def extract_text_from_image(image, binary_mode = False, lang='vie'):
	# Convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect
	if binary_mode:
		_input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		kernel3 = np.ones((3, 3), np.uint8)
		kernel5 = np.ones((5, 5), np.uint8)
		kernel7 = np.ones((7, 7), np.uint8)

		# cv2.imshow('Input', _input)
		# cv2.imshow('Erosion', img_erosion)
		# cv2.imshow('Dilation', img_dilation)
		#
		# cv2.waitKey(0)
		#############################################################################
		_input = cv2.threshold(_input, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)[1]
		T = threshold_local(_input, 11, offset=10, method="gaussian")
		_input = (_input > T).astype("uint8") * 255
		#############################################################################

		# _input = cv2.threshold(_input, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)[1]
		# # _input = cv2.erode(_input, kernel3, iterations=1)
		# # _input = cv2.dilate(_input, kernel3, iterations=1)
		# T = threshold_local(_input, 11, offset=10, method="gaussian")
		# _input = (_input > T).astype("uint8") * 255
		# _input = median_filter(_input, 5)
		#
		# _input = cv2.erode(_input, kernel5, iterations=1)
		# _input = cv2.dilate(_input, kernel5, iterations=1)
		# _input = median_filter(_input, 3)


		cv2.imwrite("/home/minhpv/Desktop/pre_processing_text/%s.jpg" %(str(random.randint(1,100000000))), _input)
	else:
		_input = image

	config = '-l {lang}'.format(lang=lang)
	# cv2.imshow("g", _input)
	# cv2.waitKey(0)
	text = pytesseract.image_to_string(_input, config=config)
	lines = text.splitlines()
	text = '\n'.join(l.strip() for l in lines if l.strip())
	return _input, text


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, default='./images')
	parser.add_argument('--use_binary', type=bool, default=True)
	parser.add_argument('--output', type=str, default='./output')
	parser.add_argument('--binary_output', type=str, default='./binary')

	FLAGS = parser.parse_args()

	allow_type = ['jpg', 'png', 'JPG', 'PNG', 'JPEG', 'jpeg']
	all_images = os.listdir(FLAGS.input)
	for image in tqdm.tqdm(all_images):
		try:
			endswith = image.split('.')[-1]
			if endswith in allow_type:
				name = image.split('.')[0]
				path_to_image = os.path.join(FLAGS.input, image)
				imread = cv2.imread(path_to_image)
				output_image, text = extract_text_from_image(image=imread, binary_mode=FLAGS.use_binary)
				#
				if FLAGS.use_binary:
					check_exist(FLAGS.binary_output)
					binary_output = '{}/{}.jpg'.format(FLAGS.binary_output, name)
					cv2.imwrite(binary_output, output_image)
				check_exist(FLAGS.output)
				output_file = '{}/{}.txt'.format(FLAGS.output, name)
				with open(output_file, 'w') as f:
					print(text)
					f.write(text)
			else:
				print("----> not allow type file: {} - type {}".format(image, endswith))
		except Exception as e:
			with open('logs.txt', 'w') as f:
				f.write(str(e))
			continue


