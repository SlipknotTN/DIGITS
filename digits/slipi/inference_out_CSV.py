#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import argparse
import base64
import h5py
import logging
import numpy as np
import PIL.Image
import os
import sys

import cv2

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import digits.config
from digits import utils, log
from digits.inference.errors import InferenceError
from digits.job import Job
from digits.utils.lmdbreader import DbReader

# Import digits.config before caffe to set the path
import caffe.io
import caffe_pb2

import csv

import digits.slipi.crop

logger = logging.getLogger('digits.tools.inference')

"""
Write results to CSV files
"""
def write_to_CSV(filename, input_list, results, labels, write_top1=False) :

    #Create CSV file
    with open(filename, 'wb') as csvfile:

        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        #Write header
        labels.insert(0,'image')
        row = labels
        if (write_top1 == True):
            row.append('Top-1 Label')
        writer.writerow(row)

        #Iterate images
        with open(input_list) as infile:
            paths = infile.readlines()
        # load and resize images
        for idx, path in enumerate(paths):
            path = path.strip()
            #Kaggle template needs the filename with extension
            row = [os.path.basename(path)]
            #Write results
            row += results[idx].flatten().tolist()
            #Add TOP-1 Label
            if (write_top1 == True):
                row += [labels[results[idx].argmax()+1]]
            writer.writerow(row)

    return

"""
Perform inference on a list of images using the specified model
"""
def infer(input_list,
          output_dir,
          jobs_dir,
          model_id,
          epoch,
          batch_size,
          layers,
          gpu,
          input_is_db,
          resize,
          oversample,
          noObjectClass,
          write_top1,
          isObjectOutputID,
          actualObjectOutputID,
          isObjectOutputFlatPosition):
    """
    Perform inference on a list of images using the specified model
    """
    # job directory defaults to that defined in DIGITS config
    if jobs_dir == 'none':
        jobs_dir = digits.config.config_value('jobs_dir')

    # load model job
    model_dir = os.path.join(jobs_dir, model_id)
    assert os.path.isdir(model_dir), "Model dir %s does not exist" % model_dir
    model = Job.load(model_dir)

    # load dataset job
    dataset_dir = os.path.join(jobs_dir, model.dataset_id)
    assert os.path.isdir(dataset_dir), "Dataset dir %s does not exist" % dataset_dir
    dataset = Job.load(dataset_dir)
    for task in model.tasks:
        task.dataset = dataset

    # retrieve snapshot file
    task = model.train_task()
    snapshot_filename = None
    epoch = float(epoch)
    if epoch == -1 and len(task.snapshots):
        # use last epoch
        epoch = task.snapshots[-1][1]
        snapshot_filename = task.snapshots[-1][0]
    else:
        for f, e in task.snapshots:
            if e == epoch:
                snapshot_filename = f
                break
    if not snapshot_filename:
        raise InferenceError("Unable to find snapshot for epoch=%s" % repr(epoch))

    # retrieve image dimensions and resize mode
    image_dims = dataset.get_feature_dims()
    height = image_dims[0]
    width = image_dims[1]
    channels = image_dims[2]
    resize_mode = dataset.resize_mode if hasattr(dataset, 'resize_mode') else 'squash'

    n_input_samples = 0  # number of samples we were able to load
    input_ids = []       # indices of samples within file list
    input_data = []      # sample data

    if input_is_db:
        # load images from database
        reader = DbReader(input_list)
        for key, value in reader.entries():
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            if datum.encoded:
                s = StringIO()
                s.write(datum.data)
                s.seek(0)
                img = PIL.Image.open(s)
                img = np.array(img)
            else:
                import caffe.io
                arr = caffe.io.datum_to_array(datum)
                # CHW -> HWC
                arr = arr.transpose((1,2,0))
                if arr.shape[2] == 1:
                    # HWC -> HW
                    arr = arr[:,:,0]
                elif arr.shape[2] == 3:
                    # BGR -> RGB
                    # XXX see issue #59
                    arr = arr[:,:,[2,1,0]]
                img = arr
            input_ids.append(key)
            input_data.append(img)
            n_input_samples = n_input_samples + 1
    else:
        # load paths from file
        paths = None
        with open(input_list) as infile:
            paths = infile.readlines()
        # load and resize images
        for idx, path in enumerate(paths):
            path = path.strip()
            try:
                image = utils.image.load_image(path.strip())
                if resize:
                    image = utils.image.resize_image(
                        image,
                        height,
                        width,
                        channels=channels,
                        resize_mode=resize_mode)
                else:
                    image = utils.image.image_to_array(
                        image,
                        channels=channels)
                input_ids.append(idx)
                input_data.append(image)
                n_input_samples = n_input_samples + 1
            except utils.errors.LoadImageError as e:
                print e

    # perform inference
    visualizations = None

    #Read labels
    with open(os.path.join(dataset._dir, dataset.labels_file)) as infile:
        labels = infile.readlines()
    labels = [label.strip() for label in labels]

    #One prediction over whole image
    if (oversample == False) :

        if n_input_samples == 0:
            raise InferenceError("Unable to load any image from file '%s'" % repr(input_list))
        elif n_input_samples == 1:
            # single image inference
            outputs, visualizations = model.train_task().infer_one(
                input_data[0],
                snapshot_epoch=epoch,
                layers=layers,
                gpu=gpu,
                resize=resize)
        else:
            if layers != 'none':
                raise InferenceError("Layer visualization is not supported for multiple inference")
            outputs = model.train_task().infer_many(
                input_data,
                snapshot_epoch=epoch,
                gpu=gpu,
                resize=resize)

    #Oversample: iterate over multiple crops and get the best result
    else :

        multiCropsOutputs = []

        for singleImage in input_data :

            netInputSize = model.train_task().crop_size

            crops = digits.slipi.crop.getMultipleCrops(image=singleImage, squareSize=netInputSize, debug=False)

            outputsSingleImage = model.train_task().infer_many(
                crops,
                snapshot_epoch=epoch,
                gpu=gpu,
                resize=resize)

            #Pick the best prediction

            #Pick the softmax output
            softmaxOut = outputsSingleImage[task._caffe_net._output_list[actualObjectOutputID]]
            softmaxOut = np.reshape(softmaxOut, (softmaxOut.shape[0], softmaxOut.shape[1]))
            #Array with the max index for each class
            maxIndexForEachClass = np.argmax(softmaxOut,axis=0)

            #Max values for each class
            classMaxPredictions = []
            classMaxPredictions.append([softmaxOut[cropId][classId] for classId, cropId in enumerate(maxIndexForEachClass)])

            classMaxPredictions = classMaxPredictions[0]

            #Consider NoObjectClass special behaviour (does not work well)
            if (noObjectClass is not None):

                noObjectClassIndex = labels.index(noObjectClass)

                #We invert noObjectClass value for subsequent maximum analysis
                classMaxPredictions[noObjectClassIndex] = 1 - classMaxPredictions[noObjectClassIndex]

                #For each class pick the crop with the best confidence

                #Consider NoObject class special behaviour
                #If any object class overcome 0.7 we keep it and select that crop's prediction as output row
                #TODO: Maybe consider other fishes maxs?!
                objectClassMaxPredictions = np.delete(classMaxPredictions,noObjectClassIndex)
                if np.max(objectClassMaxPredictions) > 0.7:
                    multiCropsOutputs.append(softmaxOut[maxIndexForEachClass[np.argmax(classMaxPredictions)]])

                #Else select as output the best NoF crop prediction
                else :
                    multiCropsOutputs.append(softmaxOut[maxIndexForEachClass[noObjectClassIndex]])

            # Is fish output evaluation, if a crop has a fish we evaluate the type of fish.
            else:

                # We set the position of the isObjectOutput for correct CSV export
                noObjectClassIndex = isObjectOutputFlatPosition

                # Evaluate IsObjectOutput
                softmaxIsObj = outputsSingleImage[task._caffe_net._output_list[isObjectOutputID]]
                softmaxIsObj = np.reshape(softmaxIsObj, (softmaxIsObj.shape[0], softmaxIsObj.shape[1]))

                # Max IsObject Index (crops that most probable contains a fish)
                maxIndexForEachIsObjClass = np.argmax(softmaxIsObj, axis=0)

                # Max values for IsObjectClass (first class is IsObject, second is NoObject)
                maxIsObjectProb = softmaxIsObj[maxIndexForEachIsObjClass[0]][0]

                print("Max isFishProb: " + str(maxIsObjectProb))

                # Draw max crops (most probable IsFish)
                #cv2.imshow("crops", crops[maxIndexForEachIsObjClass[0]])
                #cv2.waitKey(0)

                # Joint evaluation of isFish probability and specific fish probability:
                # simple product Fish probability times single spec classification

                # softmaxOut is 50x7 (crops x classes), softmaxIsObj is 50x2 (crops x classes)
                # we want to multiply all thr rows (fish class probability of softmaxOut),
                # for isFish probability (first value of each row of softmaxIsObj

                isObjectReplicated = np.repeat(np.reshape(softmaxIsObj[:,0],(len(crops),1)),7,axis=1)

                mergedPrediction = softmaxOut * isObjectReplicated
                maxIndexForMergedPrediction = np.argmax(mergedPrediction, axis=0)

                # Max values for each class
                mergedClassMaxPredictions = []
                mergedClassMaxPredictions.append(
                    [mergedPrediction[cropId][classId] for classId, cropId in enumerate(maxIndexForMergedPrediction)])

                # 1D array
                mergedClassMaxPredictions = mergedClassMaxPredictions[0]

                print ("Max merged prediction: " + str(np.max(mergedClassMaxPredictions)))

                isFishProbMergedPred = softmaxIsObj[maxIndexForMergedPrediction[np.argmax(mergedClassMaxPredictions)]][0]
                print ("IsFish Prob for max merged pred: " + str(isFishProbMergedPred))
                whichFishProbMergedPred = \
                    softmaxOut[maxIndexForMergedPrediction[np.argmax(mergedClassMaxPredictions)]][np.argmax(mergedClassMaxPredictions)]
                print ("Spec prob for max merged pred: " +
                       str(whichFishProbMergedPred))

                print ("Which fish prediction: " + labels[np.argmax(mergedClassMaxPredictions)])

                # Draw image
                # cv2.imshow("full_image",singleImage)
                # Draw max merged prediction crop
                # cv2.imshow("crops", crops[maxIndexForMergedPrediction[np.argmax(mergedClassMaxPredictions)]])
                # cv2.waitKey(0)

                # TODO: Increase the number of crops??

                # TODO: How to select the NoF prediction and other when we think that the image contains no fish?!

                # First try
                # Possible IsFish Threshold 0.9 (or even 0.99)
                # Initial condition isFishProb > 0.9, spec > 0.5

                NoFProb = 1 - isFishProbMergedPred

                if isFishProbMergedPred > 0.9 and whichFishProbMergedPred > 0.5:
                    # The sum will be slightly over 1, this should be ok for the challenge
                    specProbs = softmaxOut[maxIndexForMergedPrediction[np.argmax(mergedClassMaxPredictions)]]

                else:
                    # NoF Prob will be predominant, Fish probability is equally distributed
                    # TODO: We could make NoF more predominant
                    # TODO: Manage the case where NoF is low, but whichFish is undecided, NoF shouldn't be predominant
                    specProbs = np.ones(len(softmaxOut[0])) * isFishProbMergedPred / len(softmaxOut[0])

                # TODO: Try other epochs

                # TODO: Try OpenCV objectness to get better crops (multiscale)

                # TODO: Train FishNoFish with other datasets (e.g. persons, boats, sea without fishes)

                # TODO: Check manually dataset, remove errors (e.g. small fish from NoFish)

                specProbs = np.insert(specProbs, noObjectClassIndex, NoFProb)
                multiCropsOutputs.append(specProbs)

    if oversample == False :
        predictions = outputs[task._caffe_net._output_list[0]]
    else :
        predictions = multiCropsOutputs
        # In case of separate isObjectOutput, modify the labels (CSV header)
        if noObjectClass is None:
            labels.insert(noObjectClassIndex,'NoF')


    #Write to CSV (Kaggle Template, but very generic)

    #Create directory if not exists
    if (os.path.exists(output_dir) == False) :
        os.makedirs(output_dir)

    if (write_top1) :
        filename = 'inference_withtop1.csv'
    else :
        filename = 'inference.csv'
    write_to_CSV(os.path.join(output_dir, filename), input_list, predictions,
                 labels, write_top1=write_top1)

    # write visualization data
    if visualizations is not None and len(visualizations)>0:
        db_layers = db.create_group("layers")
        for idx, layer in enumerate(visualizations):
            vis = layer['vis'] if layer['vis'] is not None else np.empty(0)
            dset = db_layers.create_dataset(str(idx), data=vis)
            dset.attrs['name'] = layer['name']
            dset.attrs['vis_type'] = layer['vis_type']
            if 'param_count' in layer:
                dset.attrs['param_count'] = layer['param_count']
            if 'layer_type' in layer:
                dset.attrs['layer_type'] = layer['layer_type']
            dset.attrs['shape'] = layer['data_stats']['shape']
            dset.attrs['mean'] = layer['data_stats']['mean']
            dset.attrs['stddev'] = layer['data_stats']['stddev']
            dset.attrs['histogram_y'] = layer['data_stats']['histogram'][0]
            dset.attrs['histogram_x'] = layer['data_stats']['histogram'][1]
            dset.attrs['histogram_ticks'] = layer['data_stats']['histogram'][2]

    #db.close()
    #logger.info('Saved data to %s', db_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference tool - DIGITS')

    ### Positional arguments

    parser.add_argument(
        'input_list',
        help='An input file containing paths to input data')
    parser.add_argument(
        'output_dir',
        help='Directory to write outputs to')
    parser.add_argument(
        'model',
        help='Model ID')

    ### Optional arguments
    parser.add_argument(
        '-e',
        '--epoch',
        default='-1',
        help="Epoch (-1 for last)"
        )

    parser.add_argument(
        '-j',
        '--jobs_dir',
        default='none',
        help='Jobs directory (default: from DIGITS config)',
        )

    parser.add_argument(
        '-l',
        '--layers',
        default='none',
        help='Which layers to write to output ("none" [default] or "all")',
        )

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=1,
        help='Batch size',
        )

    parser.add_argument(
        '-g',
        '--gpu',
        type=int,
        default=None,
        help='GPU to use (as in nvidia-smi output, default: None)',
        )

    parser.add_argument(
        '--db',
        action='store_true',
        help='Input file is a database',
        )

    parser.add_argument(
        '--resize',
        dest='resize',
        action='store_true')

    parser.add_argument(
        '--no-resize',
        dest='resize',
        action='store_false')

    parser.add_argument(
        '--oversample',
        action='store_true',
        dest='oversample',
        help='Predictions over multiple crops for each image',
        )

    parser.add_argument(
        '--no-object-class',
        type=str,
        default=None,
        dest='no_object_class',
        help='No object class name'
        )

    parser.add_argument(
        '--write-top1',
        action='store_true',
        dest='write_top1',
        help='Add column Top1 label to output CSV',
        )

    parser.add_argument(
        '--is-object-output',
        type=int,
        default=None,
        dest='is-object-output',
        help='Output ID for the IsObject boolean evaluation',
        )

    parser.add_argument(
        '--actual-object-output',
        type=int,
        default=0,
        dest='actual-object-output',
        help='Output ID for the actual object classification',
        )

    parser.add_argument(
        '--is-object-output-flat-position',
        type=int,
        default=None,
        dest='is-object-output-flat-position',
        help='Is Object output position on the predictions export, used with isObjectOutput',
        )

    parser.set_defaults(resize=True)

    args = vars(parser.parse_args())

    if (args['oversample'] == True and args['resize'] == True) :
        raise Exception("With oversample you must select no-resize")

    try:
        infer(
            args['input_list'],
            args['output_dir'],
            args['jobs_dir'],
            args['model'],
            args['epoch'],
            args['batch_size'],
            args['layers'],
            args['gpu'],
            args['db'],
            args['resize'],
            args['oversample'],
            args['no_object_class'],
            args['write_top1'],
            args['is-object-output'],
            args['actual-object-output'],
            args['is-object-output-flat-position']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
