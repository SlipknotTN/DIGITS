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

logger = logging.getLogger('digits.tools.inference')

"""
Write results to CSV files
"""
def write_to_CSV(filename, input_list, results, labelfile, write_top1=False) :

    #Create CSV file
    with open(filename, 'wb') as csvfile:

        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        #Write header
        with open(labelfile) as infile:
            labels = infile.readlines()
        labels.insert(0,'image')
        row = [label.strip() for label in labels]
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
          write_top1):
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
    predictions = []

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

    # write to hdf5 file
    # db_path = os.path.join(output_dir, 'inference.hdf5')
    # db = h5py.File(db_path, 'w')
    #
    # # write input paths and images to database
    # db.create_dataset("input_ids", data = input_ids)
    # db.create_dataset("input_data", data = input_data)
    #
    # # write outputs to database
    # db_outputs = db.create_group("outputs")
    # for output_id, output_name in enumerate(outputs.keys()):
    #     output_data = outputs[output_name]
    #     output_key = base64.urlsafe_b64encode(str(output_name))
    #     dset = db_outputs.create_dataset(output_key, data=output_data)
    #     # add ID attribute so outputs can be sorted in
    #     # the order they appear in here
    #     dset.attrs['id'] = output_id

    #Write to CSV (Kaggle Template, but very generic)

    #Create directory if not exists
    if (os.path.exists(output_dir) == False) :
        os.makedirs(output_dir)

    if (write_top1) :
        filename = 'inference_withtop1.csv'
    else :
        filename = 'inference.csv'
    write_to_CSV(os.path.join(output_dir, filename), input_list, outputs[task._caffe_net._output_list[0]],
                 os.path.join(dataset._dir, dataset.labels_file), write_top1=write_top1)

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
        '--write-top1',
        action='store_true',
        dest='write_top1',
        help='Add column Top1 label to output CSV',
        )

    parser.set_defaults(resize=True)

    args = vars(parser.parse_args())

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
            args['write_top1']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
