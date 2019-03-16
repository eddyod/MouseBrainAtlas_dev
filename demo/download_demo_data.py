#! /usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import execute_command, create_if_not_exists
from metadata import *
from data_manager import relative_to_local

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='This script downloads input data for demo.')

parser.add_argument("-d", "--demo_data_dir", type=str, help="Directory to store demo input data")
args = parser.parse_args()

if args.demo_data_dir is None:
    demo_data_dir = DATA_ROOTDIR
else:
    demo_data_dir = args.demo_data_dir

def download_to_demo(fp):
    """
    Args:
	fp (str): file path relative to data root.
    """
    create_if_not_exists(demo_data_dir)
    s3_http_prefix = 'https://s3-us-west-1.amazonaws.com/v0.2-required-data/'
    url = s3_http_prefix + fp
    demo_fp = os.path.join(demo_data_dir, fp)
    execute_command('wget -N -P \"%s\" \"%s\"' % (os.path.dirname(demo_fp), url))
    return demo_fp


# Download raw JPEG2000 images
for img_name in [
'MD662&661-F68-2017.06.06-07.39.27_MD661_2_0203',
'MD662&661-F73-2017.06.06-09.53.20_MD661_1_0217',
'MD662&661-F79-2017.06.06-11.52.28_MD661_1_0235',
'MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250',
'MD662&661-F89-2017.06.06-16.49.49_MD661_1_0265',
'MD662&661-F94-2017.06.06-19.01.05_MD661_1_0280',
'MD662&661-F99-2017.06.06-21.14.03_MD661_1_0295',
]:

    download_to_demo(os.path.join('jp2_files', 'DEMO998', img_name + '_lossless.jp2'))
    #pass

# Download mxnet model

model_dir_name = 'inception-bn-blue'

fp = os.path.join(MXNET_MODEL_ROOTDIR, model_dir_name, 'inception-bn-blue-0000.params')
download_to_demo(relative_to_local(fp, local_root=DATA_ROOTDIR))

fp = os.path.join(MXNET_MODEL_ROOTDIR, model_dir_name, 'inception-bn-blue-symbol.json')
download_to_demo(relative_to_local(fp, local_root=DATA_ROOTDIR))

fp = os.path.join(MXNET_MODEL_ROOTDIR, model_dir_name, 'mean_224.npy')
download_to_demo(relative_to_local(fp, local_root=DATA_ROOTDIR))

# Download warp/crop operation configs.
for fn in [
'crop_orig_template',
'from_aligned_to_none',
'from_aligned_to_padded',
'from_none_to_aligned_template',
'from_none_to_padded',
'from_none_to_wholeslice',
'from_padded_to_brainstem_template',
'from_padded_to_wholeslice_template',
'from_padded_to_none',
'from_wholeslice_to_brainstem'
]:
    download_to_demo(os.path.join('operation_configs', fn + '.ini'))

# Download brain meta data
print("Download brain DEMO998 meta data")
download_to_demo(os.path.join('brains_info', 'DEMO998.ini'))

download_to_demo(os.path.join('CSHL_data_processed', 'DEMO998', 'DEMO998_sorted_filenames.txt'))
download_to_demo(os.path.join('CSHL_data_processed', 'DEMO998', 'DEMO998_prep2_sectionLimits.ini'))

# Elastix intra-stack registration parameters
download_to_demo('elastix_parameters', 'Parameters_Rigid_MutualInfo_noNumberOfSpatialSamples_4000Iters.txt')

# Rough global transform
download_to_demo(os.path.join('CSHL_simple_global_registration', 'DEMO998_registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners.json'))
download_to_demo(os.path.join('CSHL_simple_global_registration', 'DEMO998_T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol.txt'))
