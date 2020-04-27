# Lint as: python2, python3
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
"""Tests for tfx.tools.cli.handler.beam_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import click
import mock
import tensorflow as tf

from tfx.tools.cli import labels
from tfx.tools.cli.handler import beam_handler


def _MockSubprocess(cmd, env):  # pylint: disable=invalid-name, unused-argument
  # Store pipeline_args in a json file
  pipeline_args_path = env[labels.TFX_JSON_EXPORT_PIPELINE_ARGS_PATH]
  pipeline_name = 'chicago_taxi_beam'
  pipeline_root = os.path.join(os.environ['HOME'], 'tfx', 'pipelines',
                               pipeline_name)
  pipeline_args = {
      'pipeline_name': pipeline_name,
      'pipeline_root': pipeline_root
  }
  with open(pipeline_args_path, 'w') as f:
    json.dump(pipeline_args, f)
  return 0


def _MockSubprocess2(cmd, env):  # pylint: disable=invalid-name, unused-argument
  # Store pipeline_args in a json file
  pipeline_args_path = env[labels.TFX_JSON_EXPORT_PIPELINE_ARGS_PATH]
  pipeline_args = {}
  with open(pipeline_args_path, 'w') as f:
    json.dump(pipeline_args, f)
  return 0


def _MockSubprocess3(cmd, env):  # pylint: disable=unused-argument
  click.echo(cmd)
  return 0


class BeamHandlerTest(tf.test.TestCase):

  def setUp(self):
    super(BeamHandlerTest, self).setUp()
    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')
    self._tmp_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                                   self.get_temp_dir())
    self._home = os.path.join(self._tmp_dir, self._testMethodName)
    self._olddir = os.getcwd()
    os.chdir(self._tmp_dir)
    self._original_home_value = os.environ.get('HOME', '')
    os.environ['HOME'] = self._home
    self._original_beam_home_value = os.environ.get('BEAM_HOME', '')
    os.environ['BEAM_HOME'] = os.path.join(os.environ['HOME'], 'beam')
    self._beam_home = os.environ['BEAM_HOME']

    # Flags for handler.
    self.engine = 'beam'
    self.pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                      'test_pipeline_beam_1.py')
    self.pipeline_name = 'chicago_taxi_beam'
    self.pipeline_root = os.path.join(self._home, 'tfx', 'pipelines',
                                      self.pipeline_name)
    self.run_id = 'dummyID'

    # Pipeline args for mocking subprocess
    self.pipeline_args = {
        'pipeline_name': 'chicago_taxi_beam',
        'pipeline_dsl_path': self.pipeline_path
    }

  def tearDown(self):
    super(BeamHandlerTest, self).tearDown()
    if self._home:
      os.environ['HOME'] = self._original_home_value
    if self._beam_home:
      os.environ['BEAM_HOME'] = self._original_beam_home_value
    os.chdir(self._olddir)

  @mock.patch('subprocess.call', _MockSubprocess)
  def testSavePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    pipeline_args = handler._extract_pipeline_args()
    handler._save_pipeline(pipeline_args)
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler._handler_home_dir,
                         self.pipeline_args[labels.PIPELINE_NAME])))

  @mock.patch('subprocess.call', _MockSubprocess)
  def testCreatePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()
    handler_pipeline_path = os.path.join(
        handler._handler_home_dir, self.pipeline_args[labels.PIPELINE_NAME], '')
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))

  @mock.patch('subprocess.call', _MockSubprocess)
  def testCreatePipelineExistentPipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()
    # Run create_pipeline again to test.
    with self.assertRaises(SystemExit) as err:
      handler.create_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" already exists.'.format(
            self.pipeline_args[labels.PIPELINE_NAME]))

  @mock.patch('subprocess.call', _MockSubprocess)
  def testUpdatePipeline(self):
    # First create pipeline with test_pipeline.py
    pipeline_path_1 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_beam_1.py')
    flags_dict_1 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_1
    }
    handler = beam_handler.BeamHandler(flags_dict_1)
    handler.create_pipeline()

    # Update test_pipeline and run update_pipeline
    pipeline_path_2 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_beam_2.py')
    flags_dict_2 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_2
    }
    handler = beam_handler.BeamHandler(flags_dict_2)
    handler.update_pipeline()
    handler_pipeline_path = os.path.join(
        handler._handler_home_dir, self.pipeline_args[labels.PIPELINE_NAME], '')
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))

  @mock.patch('subprocess.call', _MockSubprocess)
  def testUpdatePipelineNoPipeline(self):
    # Update pipeline without creating one.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.update_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            self.pipeline_args[labels.PIPELINE_NAME]))

  @mock.patch('subprocess.call', _MockSubprocess)
  def testCompilePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.compile_pipeline()
    self.assertIn('Pipeline compiled successfully', captured.contents())

  @mock.patch('subprocess.call', _MockSubprocess2)
  def testCompilePipelineNoPipelineArgs(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.compile_pipeline()
    self.assertEqual(
        str(err.exception),
        'Unable to compile pipeline. Check your pipeline dsl.')

  @mock.patch('subprocess.call', _MockSubprocess)
  def testDeletePipeline(self):
    # First create a pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()

    # Now delete the pipeline created aand check if pipeline folder is deleted.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.delete_pipeline()
    handler_pipeline_path = os.path.join(
        handler._handler_home_dir, self.pipeline_args[labels.PIPELINE_NAME], '')
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))

  @mock.patch('subprocess.call', _MockSubprocess)
  def testDeletePipelineNonExistentPipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.delete_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            flags_dict[labels.PIPELINE_NAME]))

  def testListPipelinesNonEmpty(self):
    # First create two pipelines in the dags folder.
    handler_pipeline_path_1 = os.path.join(os.environ['BEAM_HOME'],
                                           'pipeline_1')
    handler_pipeline_path_2 = os.path.join(os.environ['BEAM_HOME'],
                                           'pipeline_2')
    tf.io.gfile.makedirs(handler_pipeline_path_1)
    tf.io.gfile.makedirs(handler_pipeline_path_2)

    # Now, list the pipelines
    flags_dict = {labels.ENGINE_FLAG: self.engine}
    handler = beam_handler.BeamHandler(flags_dict)

    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_pipelines()
    self.assertIn('pipeline_1', captured.contents())
    self.assertIn('pipeline_2', captured.contents())

  def testListPipelinesEmpty(self):
    flags_dict = {labels.ENGINE_FLAG: self.engine}
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_pipelines()
    self.assertIn('No pipelines to display.', captured.contents())

  @mock.patch('subprocess.call', _MockSubprocess)
  def testPipelineSchemaNoPipelineRoot(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()

    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.get_schema()
    self.assertEqual(
        str(err.exception),
        'Create a run before inferring schema. If pipeline is already running, then wait for it to successfully finish.'
    )

  @mock.patch('subprocess.call', _MockSubprocess)
  def testPipelineSchemaNoSchemaGenOutput(self):
    # First create a pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()

    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
    }
    handler = beam_handler.BeamHandler(flags_dict)
    tf.io.gfile.makedirs(self.pipeline_root)
    with self.assertRaises(SystemExit) as err:
      handler.get_schema()
    self.assertEqual(
        str(err.exception),
        'Either SchemaGen component does not exist or pipeline is still running. If pipeline is running, then wait for it to successfully finish.'
    )

  @mock.patch('subprocess.call', _MockSubprocess)
  def testPipelineSchemaSuccessfulRun(self):
    # First create a pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()

    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
    }
    handler = beam_handler.BeamHandler(flags_dict)
    # Create fake schema in pipeline root.
    schema_path = os.path.join(self.pipeline_root, 'SchemaGen', 'schema', '3')
    tf.io.gfile.makedirs(schema_path)
    with open(os.path.join(schema_path, 'schema.pbtxt'), 'w') as f:
      f.write('SCHEMA')
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.get_schema()
      curr_dir_path = os.path.join(os.getcwd(), 'schema.pbtxt')
      self.assertIn('Path to schema: {}'.format(curr_dir_path),
                    captured.contents())
      self.assertIn(
          '*********SCHEMA FOR {}**********'.format(self.pipeline_name.upper()),
          captured.contents())
      self.assertTrue(tf.io.gfile.exists(curr_dir_path))

  @mock.patch('subprocess.call', _MockSubprocess3)
  def testCreateRun(self):
    # Create a pipeline in dags folder.
    handler_pipeline_path = os.path.join(
        os.environ['BEAM_HOME'], self.pipeline_args[labels.PIPELINE_NAME])
    tf.io.gfile.makedirs(handler_pipeline_path)
    with open(os.path.join(handler_pipeline_path, 'pipeline_args.json'),
              'w') as f:
      json.dump(self.pipeline_args, f)

    # Now run the pipeline
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.create_run()
    self.assertIn("['%s', '%s']" % (sys.executable, self.pipeline_path),
                  captured.contents())

  def testCreateRunNoPipeline(self):
    # Run pipeline without creating one.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.create_run()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            flags_dict[labels.PIPELINE_NAME]))

  def testDeleteRun(self):
    # Create a pipeline in beam home.
    handler_pipeline_path = os.path.join(
        os.environ['BEAM_HOME'], self.pipeline_args[labels.PIPELINE_NAME])
    tf.io.gfile.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {labels.ENGINE_FLAG: self.engine, labels.RUN_ID: self.run_id}
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.delete_run()
    self.assertIn('Not supported for Beam.', captured.contents())

  def testTerminateRun(self):
    # Create a pipeline in beam home.
    handler_pipeline_path = os.path.join(
        os.environ['BEAM_HOME'], self.pipeline_args[labels.PIPELINE_NAME])
    tf.io.gfile.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {labels.ENGINE_FLAG: self.engine, labels.RUN_ID: self.run_id}
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.terminate_run()
    self.assertIn('Not supported for Beam.', captured.contents())

  def testListRuns(self):
    # Create a pipeline in beam home.
    handler_pipeline_path = os.path.join(
        os.environ['BEAM_HOME'], self.pipeline_args[labels.PIPELINE_NAME])
    tf.io.gfile.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {labels.ENGINE_FLAG: self.engine, labels.RUN_ID: self.run_id}
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_runs()
    self.assertIn('Not supported for Beam.', captured.contents())

  def testGetRun(self):
    # Create a pipeline in beam home.
    handler_pipeline_path = os.path.join(
        os.environ['BEAM_HOME'], self.pipeline_args[labels.PIPELINE_NAME])
    tf.io.gfile.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {labels.ENGINE_FLAG: self.engine, labels.RUN_ID: self.run_id}
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.get_run()
    self.assertIn('Not supported for Beam.', captured.contents())


if __name__ == '__main__':
  tf.test.main()
