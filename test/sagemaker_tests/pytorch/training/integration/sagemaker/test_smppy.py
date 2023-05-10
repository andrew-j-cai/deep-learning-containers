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

import os
import time

import boto3
import pytest
from sagemaker import utils
from sagemaker.debugger import ProfilerConfig, SMProfile

from ...integration import DEFAULT_TIMEOUT, smppy_mnist_script, training_dir
from ...integration.sagemaker.timeout import timeout
from . import invoke_pytorch_estimator
from .test_pytorchddp import validate_or_skip_pytorchddp

INSTANCE_TYPE = "ml.g4dn.12xlarge"


@pytest.mark.usefixtures("feature_smppy_present")
@pytest.mark.processor("gpu")
@pytest.mark.integration("smppy")
@pytest.mark.model("mnist")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_training_smppy(framework_version, ecr_image, sagemaker_regions):
    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator_parameters = {
            "entry_point": smppy_mnist_script,
            "role": "SageMakerRole",
            "instance_count": 1,
            "instance_type": INSTANCE_TYPE,
            "framework_version": framework_version,
            "hyperparameters": {"epochs": 1},
            "profiler_config": ProfilerConfig(
                smprofile_params=SMProfile(cpu_profiling_duration=3600)
            ),
            "debug_hook_config": False,
        }
        upload_s3_data_args = {"path": training_dir, "key_prefix": "pytorch/mnist"}
        job_name = utils.unique_name_from_base("test-pt-smppy-training")
        pytorch, _ = invoke_pytorch_estimator(
            ecr_image,
            sagemaker_regions,
            estimator_parameters,
            upload_s3_data_args=upload_s3_data_args,
            job_name=job_name,
        )
        _check_and_cleanup_s3_output(pytorch, 40)


@pytest.mark.usefixtures("feature_smppy_present")
@pytest.mark.processor("gpu")
@pytest.mark.integration("smppy")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_training_smppy_distributed(framework_version, ecr_image, sagemaker_regions):
    with timeout(minutes=DEFAULT_TIMEOUT):
        validate_or_skip_pytorchddp(ecr_image)
        distribution = {"pytorchddp": {"enabled": True}}
        estimator_parameters = {
            "entry_point": smppy_mnist_script,
            "role": "SageMakerRole",
            "instance_count": 2,
            "instance_type": INSTANCE_TYPE,
            "framework_version": framework_version,
            "distribution": distribution,
            "hyperparameters": {"epochs": 1},
            "profiler_config": ProfilerConfig(
                smprofile_params=SMProfile(cpu_profiling_duration=3600)
            ),
            "debug_hook_config": False,
        }
        upload_s3_data_args = {"path": training_dir, "key_prefix": "pytorch/mnist"}
        job_name = utils.unique_name_from_base("test-pt-smppy-training")
        pytorch, _ = invoke_pytorch_estimator(
            ecr_image,
            sagemaker_regions,
            estimator_parameters,
            upload_s3_data_args=upload_s3_data_args,
            job_name=job_name,
        )
        _check_and_cleanup_s3_output(pytorch, 60)


def _check_and_cleanup_s3_output(estimator, wait_time):
    s3 = boto3.client("s3")
    bucket = estimator.output_path.replace("s3://", "").rstrip("/")

    # Give postprocessing rule some time to complete
    time.sleep(wait_time)

    postproc_contents = s3.list_objects_v2(
        Bucket=bucket, Prefix=_get_detailed_profiler_output_prefix(estimator)
    ).get("Contents")
    assert len(postproc_contents) > 0
    for file in postproc_contents:
        assert file.get("Size") > 0

    all_contents = s3.list_objects_v2(
        Bucket=bucket, Prefix=os.path.join(estimator.latest_training_job.name, "")
    ).get("Contents")
    for file in all_contents:
        s3.delete_object(Bucket=bucket, Key=file["Key"])


def _get_detailed_profiler_output_prefix(estimator):
    config_name = None
    for processing in estimator.profiler_rule_configs:
        params = processing.get("RuleParameters", dict())
        rule = config_name = params.get("rule_to_invoke", "")
        if rule == "DetailedProfilerProcessing":
            config_name = processing.get("RuleConfigurationName")
            break
    return os.path.join(
        estimator.latest_training_job.name,
        "rule-output",
        config_name,
        "",
    )