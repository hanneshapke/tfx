# Copyright 2019 Google LLC. All Rights Reserved.
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
"""E2E Airflow tests for CLI."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import locale
import os
import subprocess
import tempfile

from click import testing as click_testing
import tensorflow as tf

from tfx.tools.cli.cli_main import cli_group
from tfx.utils import io_utils


class CliAirflowEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(CliAirflowEndToEndTest, self).setUp()

    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'

    # Setup airflow_home in a temp directory
    self._airflow_home = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', tempfile.mkdtemp()),
        self._testMethodName)
    self._old_airflow_home = os.environ.get('AIRFLOW_HOME')
    os.environ['AIRFLOW_HOME'] = self._airflow_home
    self._old_home = os.environ.get('HOME')
    os.environ['HOME'] = self._airflow_home
    tf.logging.info('Using %s as AIRFLOW_HOME and HOME in this e2e test',
                    self._airflow_home)

    # Testdata path.
    self._testdata_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    # Copy data.
    chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'examples', 'chicago_taxi_pipeline', '')
    data_dir = os.path.join(chicago_taxi_pipeline_dir, 'data', 'simple')
    content = tf.gfile.ListDirectory(data_dir)
    assert content, 'content in {} is empty'.format(data_dir)
    target_data_dir = os.path.join(self._airflow_home, 'taxi', 'data', 'simple')
    io_utils.copy_dir(data_dir, target_data_dir)
    assert tf.gfile.IsDirectory(target_data_dir)
    content = tf.gfile.ListDirectory(target_data_dir)
    assert content, 'content in {} is {}'.format(target_data_dir, content)
    io_utils.copy_file(
        os.path.join(chicago_taxi_pipeline_dir, 'taxi_utils.py'),
        os.path.join(self._airflow_home, 'taxi', 'taxi_utils.py'))

    # Initialize database.
    _ = subprocess.check_output(['airflow', 'initdb'])

    # Initialize CLI runner.
    self.runner = click_testing.CliRunner()

    # Start scheduler.
    self._scheduler = subprocess.Popen(['airflow', 'scheduler'])

  def tearDown(self):
    super(CliAirflowEndToEndTest, self).tearDown()
    if self._old_airflow_home:
      os.environ['AIRFLOW_HOME'] = self._old_airflow_home
    if self._old_home:
      os.environ['HOME'] = self._old_home
    if self._scheduler:
      self._scheduler.terminate()

  def _valid_create_and_check(self, pipeline_path, pipeline_name):
    handler_pipeline_path = os.path.join(self._airflow_home, 'dags',
                                         pipeline_name)

    # Create a pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'airflow', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating pipeline', result.output)
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))
    self.assertIn('Pipeline {} created successfully.'.format(pipeline_name),
                  result.output)

  def test_pipeline_create_auto_detect(self):
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')
    pipeline_name = 'chicago_taxi_simple'
    handler_pipeline_path = os.path.join(self._airflow_home, 'dags',
                                         pipeline_name)
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'auto', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating pipeline', result.output)
    self.assertIn('Detected Airflow', result.output)
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))
    self.assertIn('Pipeline {} created successfully.'.format(pipeline_name),
                  result.output)

  def test_pipeline_create(self):
    # Create a pipeline.
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')
    pipeline_name = 'chicago_taxi_simple'
    self._valid_create_and_check(pipeline_path, pipeline_name)

    # Test pipeline create when pipeline already exists.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'create', '--engine', 'airflow', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating pipeline', result.output)
    self.assertTrue('Pipeline {} already exists.'.format(pipeline_name),
                    result.output)

  def test_pipeline_update(self):
    pipeline_name = 'chicago_taxi_simple'
    handler_pipeline_path = os.path.join(self._airflow_home, 'dags',
                                         pipeline_name)

    # Try pipeline update when pipeline does not exist.
    pipeline_path_1 = os.path.join(self._testdata_dir,
                                   'test_pipeline_airflow_1.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'update', '--engine', 'airflow', '--pipeline_path',
        pipeline_path_1
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Updating pipeline', result.output)
    self.assertIn('Pipeline {} does not exist.'.format(pipeline_name),
                  result.output)
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))

    # Now update an existing pipeline.
    self._valid_create_and_check(pipeline_path_1, pipeline_name)

    pipeline_path_2 = os.path.join(self._testdata_dir,
                                   'test_pipeline_airflow_2.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'update', '--engine', 'airflow', '--pipeline_path',
        pipeline_path_2
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Updating pipeline', result.output)
    self.assertIn('Pipeline {} updated successfully.'.format(pipeline_name),
                  result.output)
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))

  def test_pipeline_compile(self):
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_2.py')
    result = self.runner.invoke(cli_group, [
        'pipeline', 'compile', '--engine', 'airflow', '--pipeline_path',
        pipeline_path
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Compiling pipeline', result.output)
    self.assertIn('Pipeline compiled successfully', result.output)

  def test_pipeline_delete(self):
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')
    pipeline_name = 'chicago_taxi_simple'
    handler_pipeline_path = os.path.join(self._airflow_home, 'dags',
                                         pipeline_name)

    # Try deleting a non existent pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'delete', '--engine', 'airflow', '--pipeline_name',
        pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Deleting pipeline', result.output)
    self.assertIn('Pipeline {} does not exist.'.format(pipeline_name),
                  result.output)
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))

    # Create a pipeline.
    self._valid_create_and_check(pipeline_path, pipeline_name)

    # Now delete the pipeline.
    result = self.runner.invoke(cli_group, [
        'pipeline', 'delete', '--engine', 'airflow', '--pipeline_name',
        pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Deleting pipeline', result.output)
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))
    self.assertIn('Pipeline {} deleted successfully.'.format(pipeline_name),
                  result.output)

  def test_pipeline_list(self):

    # Try listing pipelines when there are none.
    result = self.runner.invoke(cli_group,
                                ['pipeline', 'list', '--engine', 'airflow'])
    self.assertIn('CLI', result.output)
    self.assertIn('Listing all pipelines', result.output)

    # Create pipelines.
    pipeline_name_1 = 'chicago_taxi_simple'
    pipeline_path_1 = os.path.join(self._testdata_dir,
                                   'test_pipeline_airflow_1.py')
    self._valid_create_and_check(pipeline_path_1, pipeline_name_1)

    pipeline_name_2 = 'chicago_taxi_simple_v2'
    pipeline_path_2 = os.path.join(self._testdata_dir,
                                   'test_pipeline_airflow_3.py')
    self._valid_create_and_check(pipeline_path_2, pipeline_name_2)

    # List pipelines.
    result = self.runner.invoke(cli_group,
                                ['pipeline', 'list', '--engine', 'airflow'])
    self.assertIn('CLI', result.output)
    self.assertIn('Listing all pipelines', result.output)
    self.assertIn(pipeline_name_1, result.output)
    self.assertIn(pipeline_name_2, result.output)

  def _valid_run_and_check(self, pipeline_name):
    result = self.runner.invoke(cli_group, [
        'run', 'create', '--engine', 'airflow', '--pipeline_name', pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating a run for pipeline: {}'.format(pipeline_name),
                  result.output)
    self.assertNotIn('Pipeline {} does not exist.'.format(pipeline_name),
                     result.output)
    self.assertIn('Run created for pipeline: {}'.format(pipeline_name),
                  result.output)

  def test_run_create(self):
    pipeline_name = 'chicago_taxi_simple'
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')

    # Try running a non-existent pipeline.
    result = self.runner.invoke(cli_group, [
        'run', 'create', '--engine', 'airflow', '--pipeline_name', pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating a run for pipeline: {}'.format(pipeline_name),
                  result.output)
    self.assertIn('Pipeline {} does not exist.'.format(pipeline_name),
                  result.output)

    # Now create a pipeline.
    self._valid_create_and_check(pipeline_path, pipeline_name)

    # Run pipeline.
    self._valid_run_and_check(pipeline_name)

  def test_run_list(self):
    pipeline_name = 'chicago_taxi_simple'
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_airflow_1.py')

    # Now create a pipeline.
    self._valid_create_and_check(pipeline_path, pipeline_name)

    # Run pipeline.
    self._valid_run_and_check(pipeline_name)

    # Run pipeline.
    self._valid_run_and_check(pipeline_name)

    # List pipeline runs.
    result = self.runner.invoke(cli_group, [
        'run', 'list', '--engine', 'airflow', '--pipeline_name', pipeline_name
    ])
    self.assertIn('CLI', result.output)
    self.assertIn('Listing all runs of pipeline: {}'.format(pipeline_name),
                  result.output)

  def test_uninstalled_orchestrator_kubeflow(self):
    result = self.runner.invoke(cli_group,
                                ['pipeline', 'list', '--engine', 'kubeflow'])
    self.assertIn('CLI', result.output)
    self.assertIn('Listing all pipelines', result.output)
    self.assertIn('Kubeflow not found', result.output)

  def test_incorrect_runner_airflow(self):
    pipeline_path = os.path.join(self._testdata_dir,
                                 'test_pipeline_kubeflow_1.py')
    result = self.runner.invoke(
        cli_group, ['pipeline', 'create', '--pipeline_path', pipeline_path])
    self.assertIn('CLI', result.output)
    self.assertIn('Creating pipeline', result.output)
    self.assertIn('airflow runner not found in dsl.', result.output)


if __name__ == '__main__':
  tf.test.main()