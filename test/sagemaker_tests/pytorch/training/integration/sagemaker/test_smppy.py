# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import pytest
from sagemaker import utils
from sagemaker.debugger import ProfilerConfig, SMProfile
from sagemaker.pytorch import PyTorch

from ...integration import training_dir, smppy_mnist_script, DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout
from . import invoke_pytorch_estimator

INSTANCE_TYPE = 'ml.g4dn.12xlarge'


@pytest.mark.usefixtures("feature_smppy_present")
@pytest.mark.processor("gpu")
@pytest.mark.integration("smppy")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_training_smppy(framework_version, ecr_image, sagemaker_regions):
    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator_parameters = {
            'entry_point': smppy_mnist_script,
            'role': 'arn:aws:iam::920076894685:role/SageMakerRole',
            'instance_count': 2,
            'instance_type': INSTANCE_TYPE,
            'framework_version': framework_version,
            'hyperparameters': {'epochs': 1},
            'profiler_config': ProfilerConfig(
                smprofile_params = SMProfile(cpu_profiling_duration=3600)
            ),
            'debug_hook_config': False,
        }
        upload_s3_data_args = {
            'path': training_dir,
            'key_prefix': 'pytorch/mnist'
        }
        job_name=utils.unique_name_from_base('test-pt-smppy-training')
        invoke_pytorch_estimator(ecr_image, sagemaker_regions, estimator_parameters, upload_s3_data_args=upload_s3_data_args, job_name=job_name)
