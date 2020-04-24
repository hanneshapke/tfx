# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for TFX-on-KFP and GCP services."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import time
from typing import Text

from absl import logging
import kfp
import tensorflow as tf

from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_kubeflow_gcp
from tfx.orchestration import data_types
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration.kubeflow import test_utils

# The endpoint of the KFP instance.
# This test fixture assumes an established KFP instance authenticated via
# inverse proxy.
_KFP_ENDPOINT = os.environ['KFP_E2E_ENDPOINT']

# Timeout for a single pipeline run. Set to 6 hours.
_TIME_OUT_SECONDS = 21600

# The base container image name to use when building the image used in tests.
_BASE_CONTAINER_IMAGE = os.environ['KFP_E2E_BASE_CONTAINER_IMAGE']

# The project id to use to run tests.
_GCP_PROJECT_ID = os.environ['KFP_E2E_GCP_PROJECT_ID']

# The GCP region in which the end-to-end test is run.
_GCP_REGION = os.environ['KFP_E2E_GCP_REGION']

# The GCP bucket to use to write output artifacts.
_BUCKET_NAME = os.environ['KFP_E2E_BUCKET_NAME']

# Various execution status of a KFP pipeline.
_KFP_RUNNING_STATUS = 'running'
_KFP_SUCCESS_STATUS = 'succeeded'
_KFP_FAIL_STATUS = 'failed'
_KFP_SKIPPED_STATUS = 'skipped'
_KFP_ERROR_STATUS = 'error'

_KFP_FINAL_STATUS = frozenset((_KFP_SUCCESS_STATUS, _KFP_FAIL_STATUS,
                               _KFP_ERROR_STATUS, _KFP_ERROR_STATUS))

# The location of test user module file.
# It is retrieved from inside the container subject to testing.
_MODULE_FILE = '/tfx-src/tfx/examples/chicago_taxi_pipeline/taxi_utils.py'

# Parameterize worker type/count for easily ramping up the pipeline scale.
_WORKER_COUNT = data_types.RuntimeParameter(
    name='worker-count',
    default=2,
    ptype=int,
)

_WORKER_TYPE = data_types.RuntimeParameter(
    name='worker-type',
    default='standard',
    ptype=str,
)

_AI_PLATFORM_SERVING_ARGS = {
    'model_name': 'chicago_taxi',
    'project_id': _GCP_PROJECT_ID,
    'regions': [_GCP_REGION],
}

_BEAM_PIPELINE_ARGS = [
    '--runner=DataflowRunner',
    '--experiments=shuffle_mode=auto',
    '--project=' + _GCP_PROJECT_ID,
    '--temp_location=gs://' + os.path.join(_BUCKET_NAME, 'dataflow', 'tmp'),
    '--region=' + _GCP_REGION,
    '--disk_size_gb=50',
]


class KubeflowGcpPerfTest(test_utils.BaseKubeflowTest):

  @classmethod
  def setUpClass(cls):
    super(test_utils.BaseKubeflowTest, cls).setUpClass()
    # Create a container image for use by test pipelines.
    base_container_image = _BASE_CONTAINER_IMAGE

    cls._container_image = '{}:{}'.format(base_container_image,
                                          cls._random_id())
    cls._build_and_push_docker_image(cls._container_image)

  @classmethod
  def tearDownClass(cls):
    super(test_utils.BaseKubeflowTest, cls).tearDownClass()

  # TODO(jxzheng): workaround for 1hr timeout limit in kfp.Client().
  # This should be changed after
  # https://github.com/kubeflow/pipelines/issues/3630 is fixed.
  # Currently gcloud authentication token has a one hour expiration by default
  # but kfp.Client() does not have a refreshing mechanism in place. This
  # causes failure to get run response for a long pipeline execution
  # (> 1 hour).
  # Instead of implementing a full-fledged authentication refreshing mechanism
  # here. We chose re-creating kfp.Client() frequently to make sure the
  # authentication does not expire. This is based on that kfp.Client() is
  # very light-weighted.
  # See more details at
  # https://github.com/kubeflow/pipelines/issues/3630
  def _assert_successful_run_completion(self, host: Text, run_id: Text,
                                        timeout: int):
    """Waits and asserts a successful KFP pipeline execution.

    Args:
      host: the endpoint of the KFP deployment.
      run_id: the run ID of the execution, can be obtained from the respoonse
        when submitting the pipeline.
      timeout: maximal waiting time for this execution, in seconds.

    Raises:
      RuntimeError: when timeout exceeds after waiting for specified duration.
    """
    status = None
    start_time = datetime.datetime.now()
    while status is None or status.lower() not in _KFP_FINAL_STATUS:
      client = kfp.Client(host=host)
      get_run_response = client._run_api.get_run(run_id=run_id)
      # Skip transient error or unavailability.
      if get_run_response is None or get_run_response.run is None:
        logging.info('Skipped due to lack of response at %s',
                     datetime.datetime.now())
        continue
      status = get_run_response.run.status
      elapsed_time = (datetime.datetime.now() - start_time).seconds
      logging.info('Waiting for the job to complete...')
      if elapsed_time > timeout:
        raise RuntimeError('Waiting for run timeout at %s' %
                           datetime.datetime.now().strftime('%H:%M:%S'))
      time.sleep(10)

    self.assertEqual(status.lower(), _KFP_SUCCESS_STATUS)

  def _compile_and_run_pipeline(self, pipeline: tfx_pipeline.Pipeline,
                                **kwargs):
    """Compiles and runs a KFP pipeline.

    In this method, provided TFX pipeline will be submitted via kfp.Client()
    instead of from Argo.

    Args:
      pipeline: The logical pipeline to run.
      **kwargs: Key-value pairs of runtime paramters passed to the pipeline
        execution.
    """
    client = kfp.Client(host=_KFP_ENDPOINT)

    pipeline_name = pipeline.pipeline_info.pipeline_name
    config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=self._get_kubeflow_metadata_config(),
        tfx_image=self._container_image)
    kubeflow_dag_runner.KubeflowDagRunner(config=config).run(pipeline)

    file_path = os.path.join(self._test_dir, '{}.tar.gz'.format(pipeline_name))
    self.assertTrue(tf.io.gfile.exists(file_path))

    run_result = client.create_run_from_pipeline_package(
        pipeline_file=file_path, arguments=kwargs)
    run_id = run_result.run_id

    self._assert_successful_run_completion(
        host=_KFP_ENDPOINT,
        run_id=run_id,
        timeout=_TIME_OUT_SECONDS)

  def testFullTaxiGcpPipeline(self):
    pipeline_name = 'gcp-perf-test-full-e2e-test-{}'.format(self._random_id())

    # Custom CAIP training job using a testing image.
    ai_platform_training_args = {
        'project': _GCP_PROJECT_ID,
        'region': _GCP_REGION,
        'scaleTier': 'CUSTOM',
        'masterType': 'large_model',
        'masterConfig': {
            'imageUri': self._container_image
        },
        'workerType': _WORKER_TYPE,
        'parameterServerType': 'standard',
        'workerCount': _WORKER_COUNT,
        'parameterServerCount': 1
    }

    pipeline = taxi_pipeline_kubeflow_gcp.create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        module_file=_MODULE_FILE,
        ai_platform_training_args=ai_platform_training_args,
        ai_platform_serving_args=_AI_PLATFORM_SERVING_ARGS,
        beam_pipeline_args=_BEAM_PIPELINE_ARGS)
    self._compile_and_run_pipeline(pipeline=pipeline, query_sample_rate=1)


if __name__ == '__main__':
  tf.test.main()
