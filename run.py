# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

__author__ = "Marganne Louis <louis.marganne@student.uliege.be>"


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from shapely import wkt

from cytomine.models import ImageInstanceCollection, AnnotationCollection, TermCollection, Job, AttachedFile, Property
from cytomine import CytomineJob

from landmark_HM_models import UNET
from utils import *

import random
import joblib
import math
import glob
import sys


def main(argv):
	with CytomineJob.from_cli(argv) as cj:
		cj.job.update(status=Job.RUNNING, progress=0, status_comment="Initialization of the training phase...")

		## 1. Create working directories on the machine:
		# - WORKING_PATH/images: store input images
		# - WORKING_PATH/landmarks: store landmarks coordinates of input images
		# - WORKING_PATH/rescaled: store rescaled version of images/landmarks
		# - WORKING_PATH/out: store output from the model

		base_path = "{}".format(os.getenv("HOME"))
		working_path = os.path.join(base_path, str(cj.job.id))
		images_path = os.path.join(working_path, 'images/')
		landmarks_path = os.path.join(working_path, 'landmarks/')
		rescaled_path = os.path.join(working_path, 'rescaled/')
		rescaled_images_path = os.path.join(rescaled_path, 'images/')
		rescaled_landmarks_path = os.path.join(rescaled_path, 'landmarks/')
		out_path = os.path.join(working_path, 'out/')

		if not os.path.exists(working_path):
			os.makedirs(working_path)
			os.makedirs(images_path)
			os.makedirs(landmarks_path)
			os.makedirs(rescaled_path)
			os.makedirs(rescaled_images_path)
			os.makedirs(rescaled_landmarks_path)
			os.makedirs(out_path)


		## 2. Parse input data
		# Select list of terms corresponding to input
		terms_collection = TermCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
		terms_names_to_id = {term.name : term.id for term in terms_collection}

		if cj.parameters.cytomine_id_terms == 'lm':
			terms_names = [str(i).zfill(2) + '-v-lm' for i in range(1,15)] if cj.parameters.butterfly_side == 'ventral' else [str(i).zfill(2) + '-d-lm' for i in range(1,19)]
			terms_ids = [terms_names_to_id[name] for name in terms_names]
		elif cj.parameters.cytomine_id_terms == 'slm':
			terms_names = [str(i) + '-v-slm' for i in range(15,30)] if cj.parameters.butterfly_side == 'ventral' else [str(i) + '-d-slm' for i in range(19,45)]
			terms_ids = [terms_names_to_id[name] for name in terms_names]
		elif cj.parameters.cytomine_id_terms == 'all':
			terms_names = [str(i).zfill(2) + '-v-lm' for i in range(1,15)] + [str(i) + '-v-slm' for i in range(15,30)] \
			if cj.parameters.butterfly_side == 'ventral' else [str(i).zfill(2) + '-d-lm' for i in range(1,19)] + [str(i) + '-d-slm' for i in range(19,45)]
			terms_ids = [terms_names_to_id[name] for name in terms_names]
		else:
			terms_ids = [int(term_id) for term_id in cj.parameters.cytomine_id_terms.split(',')]

		# Select list of images corresponding to input
		images = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
		image_id_to_object = {image.id : image for image in images}
		if cj.parameters.cytomine_training_images == 'all':
			tr_images = images
		else:
			images_ids = [int(image_id) for image_id in cj.parameters.cytomine_training_images.split(',')]
			tr_images = [image_id_to_object[image_id] for image_id in images_ids]


		## 3. Download the images and the corresponding landmarks
		cj.job.update(progress=5, statusComment='Downloading images...')

		xpos = {}
		ypos = {}
		terms = {}

		# Download images
		for image in tr_images:
			image.download(dest_pattern=images_path+'%d.tif' % image.id)

			annotations = AnnotationCollection()
			annotations.project = cj.parameters.cytomine_id_project
			annotations.showWKT = True
			annotations.showTerm = True
			annotations.image = image.id
			annotations.fetch()

			for annotation in annotations:
				if annotation.term[0] in terms_ids:
					point = wkt.loads(annotation.location)
					(cx, cy) = point.xy
					xpos[(annotation.term[0], image.id)] = int(cx[0])
					ypos[(annotation.term[0], image.id)] = image.height - int(cy[0])
					terms[annotation.term[0]] = 1

		# Download landmarks
		for image in tr_images:
			file = open(landmarks_path+'%d.txt' % image.id, 'w')
			for term_id in terms.keys():
				if (term_id, image.id) in xpos:
					file.write('%d %d %d\n' % (
						term_id,
						xpos[(term_id, image.id)],
						ypos[(term_id, image.id)]))
			file.close()


		## 4. Apply rescale to input
		cj.job.update(progress=15, statusComment='Rescaling images...')

		org_images = glob.glob(images_path+'*.tif')
		org_lmks = glob.glob(landmarks_path+'*.txt')
		for i in range(len(org_lmks)):
			image = cv.imread(org_images[i], cv.IMREAD_UNCHANGED)
			lm = np.loadtxt(org_lmks[i])[:,1:3]
			im_name = os.path.basename(org_lmks[i])[:-4]
			re_img, re_lm = rescale_pad_img(image, lm, 256)
			cv.imwrite(rescaled_images_path+im_name+'.png', re_img)
			np.savetxt(rescaled_landmarks_path+im_name+'.txt', re_lm, fmt='%d')


		## 5. Construct training and validation set with tensorflow
		# Setup
		seed = 42

		N = len(terms_ids)

		im_path = glob.glob(rescaled_images_path+'*.png')
		lm_path = glob.glob(rescaled_landmarks_path+'*.txt')

		# Loading landmarks as list of arrays
		idxs = []
		kp_list = []
		for i in range(len(lm_path)):  
			idxs.append(i)
			kp = np.loadtxt(lm_path[i])
			kp = np.array(kp).astype(np.int32)
			kp_list.append(kp)

		# Generation of tensorflow dataset
		train_idxs, val_idxs = train_test_split(idxs, test_size=0.1, random_state=seed)
	
		train_images = [im_path[i] for i in train_idxs]
		train_lmks = [kp_list[i] for i in train_idxs]
		val_images = [im_path[i] for i in val_idxs]
		val_lmks = [kp_list[i] for i in val_idxs] 
		
		
		steps_per_epoch = math.ceil(len(train_images) / cj.parameters.model_batch_size)
		val_steps = math.ceil(len(val_images) / cj.parameters.model_batch_size)

		# Train dataset
		tr_ds = tf.data.Dataset.from_tensor_slices((train_images, train_lmks))
		tr_ds = tr_ds.map(lambda image, lm: parse_data(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		tr_ds = tr_ds.map(lambda image, lm: aug_apply(image, lm, N), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		tr_ds = tr_ds.map(lambda image, lm: normalize(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		tr_ds = tr_ds.map(lambda image, lm: to_hm(image, lm, N, cj.parameters.model_sigma, cj.parameters.model_probability_function), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		tr_ds = tr_ds.prefetch(tf.data.experimental.AUTOTUNE).shuffle(
				random.randint(0, len(im_path))).batch(cj.parameters.model_batch_size).repeat()

		# Validation dataset
		val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_lmks))
		val_ds = val_ds.map(lambda image, lm: parse_data(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)   
		val_ds = val_ds.map(lambda image, lm: normalize(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		val_ds = val_ds.map(lambda image, lm: to_hm(image, lm, N, cj.parameters.model_sigma, cj.parameters.model_probability_function), num_parallel_calls=tf.data.experimental.AUTOTUNE) 
		val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE).shuffle(
				random.randint(0, len(im_path))).batch(cj.parameters.model_batch_size).repeat()

		## 6. Train the model
		cj.job.update(progress=20, statusComment='Training...')

		# Checkpoints setup
		checkpoint_filename = out_path + '%d_model.hdf5' % cj.job.id
		checkpoint = ModelCheckpoint(checkpoint_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		callbacks = [checkpoint]

		# Train
		optim = RMSprop(learning_rate=0.001)
		rmse = tf.keras.metrics.RootMeanSquaredError()
		model = UNET(input_shape=(256,256,3), H=256, W=256, nKeypoints=N)
		model.compile(loss="mse", optimizer=optim, metrics=[rmse])
		model.fit(tr_ds, epochs=cj.parameters.model_epochs, callbacks=callbacks,
					validation_data=val_ds, steps_per_epoch=steps_per_epoch,
					validation_steps= val_steps, verbose=1)


		## 7. Save useful info for prediction
		cj.job.update(progress=95, statusComment='Uploading model...')

		# Save parameters for the prediction
		parameters_hash = {}
		parameters_hash['cytomine_id_terms'] = list(terms.keys())
		parameters_hash['butterfly_side'] = cj.parameters.butterfly_side
		parameters_hash['model_epochs'] = cj.parameters.model_epochs
		parameters_hash['model_batch_size'] = cj.parameters.model_batch_size
		parameters_hash['model_sigma'] = cj.parameters.model_sigma
		parameters_hash['model_probability_function'] = cj.parameters.model_probability_function
		parameters_hash['N'] = N

		parameters_filename = joblib.dump(parameters_hash, out_path + '%d_parameters.joblib' % cj.job.id, compress=3)[0]

		# Upload files to Cytomine
		AttachedFile(
			cj.job,
			domainIdent=cj.job.id,
			filename=checkpoint_filename,
			domainClassName='be.cytomine.processing.Job'
		).upload()
		AttachedFile(
			cj.job,
			domainIdent=cj.job.id,
			filename=parameters_filename,
			domainClassName='be.cytomine.processing.Job'
		).upload()

		cj.job.update(status=Job.TERMINATED, progress=100, statusComment='Job terminated.')

if __name__ == '__main__':
	main(sys.argv[1:])