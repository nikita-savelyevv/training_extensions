# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from subprocess import run
from copy import deepcopy

import os
import pytest

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component
from ote_cli.registry import Registry

from common import (
    create_venv,
    get_some_vars,
    ote_demo_deployment_testing,
    ote_demo_testing,
    ote_demo_openvino_testing,
    ote_deploy_openvino_testing,
    ote_eval_deployment_testing,
    ote_eval_openvino_testing,
    ote_eval_testing,
    ote_train_testing,
    ote_export_testing,
    pot_optimize_testing,
    pot_eval_testing,
    nncf_optimize_testing,
    nncf_export_testing,
    nncf_eval_testing,
    nncf_eval_openvino_testing,
    args,
    wrong_paths,
    ote_train_common
)

root = '/tmp/ote_cli/'
ote_dir = os.getcwd()

templates = Registry('external').filter(task_type='DETECTION').templates
templates_ids = [template.model_template_id for template in templates]


class TestTrainCommonDetection:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_template(self, template):
        error_string = "ote train: error: the following arguments are required: template"
        ret = ote_train_common(template, root, [])
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_train_ann_file(self, template):
        error_string = "ote train: error: the following arguments are required: --train-ann-files"
        command_line = [template.model_template_id,
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_train_data_roots(self, template):
        error_string = "ote train: error: the following arguments are required: --train-data-roots"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_val_ann_file(self, template):
        error_string = "ote train: error: the following arguments are required: --val-ann-files"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_val_data_roots(self, template):
        error_string = "ote train: error: the following arguments are required: --val-data-roots"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_save_model_to(self, template):
        error_string = "ote train: error: the following arguments are required: --save-model-to"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)
    
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_wrong_required_paths(self, template):
        error_string = "Path is not valid"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        for i in [4, 6, 8, 10, 12]:
            for case in wrong_paths.values():
                temp = deepcopy(command_line)
                temp[i] = case
                ret = ote_train_common(template, root, temp)
                assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_hpo_not_enabled(self, template):
        error_string = "Parameter --hpo-time-ratio must be used with --enable-hpo key"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        '--hpo-time-ratio', '4']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_wrong_hpo_value(self, template):
        error_string = "ote train: error: argument --hpo-time-ratio: invalid float value: 'STRING_VALUE'"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        '--enable-hpo',
                        '--hpo-time-ratio',
                        'STRING_VALUE']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_wrong_hpo_value(self, template):
        error_string = "Parameter --hpo-time-ratio must not be negative"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        '--enable-hpo',
                        '--hpo-time-ratio',
                        '-1']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)


class TestTrainDetectionTemplate:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_batch_size_type(self, template):
        error_string = "invalid int value"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.batch_size']
        cases = ["1.0", "-1", "Alpha"]
        for case in cases:
            temp = deepcopy(command_args)
            temp.append(case)
            ret = ote_train_common(template, root, temp)
            assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_batch_size_positive_case(self, template):
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.batch_size', '1']
        ret = ote_train_common(template, root, command_args)
        assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_batch_size_oob(self, template):
        error_string = "is out of bounds."
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.batch_size']
        cases = ["0", "513"]
        for case in cases:
            temp = deepcopy(command_args)
            temp.append(case)
            ret = ote_train_common(template, root, temp)
            assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_learning_rate_type(self, template):
        error_string = "invalid float value"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.learning_rate']
        cases = ["-1", "Alpha"]
        for case in cases:
            temp = deepcopy(command_args)
            temp.append(case)
            ret = ote_train_common(template, root, temp)
            assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_learning_rate_positive_case(self, template):
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.learning_rate', '0.01']
        ret = ote_train_common(template, root, command_args)
        assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_learning_rate_oob(self, template):
        error_string = "is out of bounds."
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.learning_rate']
        cases = ["0.0", "0.2"]
        for case in cases:
            temp = deepcopy(command_args)
            temp.append(case)
            ret = ote_train_common(template, root, temp)
            assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_lr_warmup_iters_type(self, template):
        error_string = "invalid int value"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.learning_rate_warmup_iters']
        cases = ["1.0", "-1", "Alpha"]
        for case in cases:
            temp = deepcopy(command_args)
            temp.append(case)
            ret = ote_train_common(template, root, temp)
            assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_learning_rate_warmup_iters_positive_case(self, template):
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.learning_rate_warmup_iters', '1']
        ret = ote_train_common(template, root, command_args)
        assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_lr_warmup_iters_oob(self, template):
        error_string = "is out of bounds."
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.learning_rate_warmup_iters',
                        '10001'
                        ]
        ret = ote_train_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_num_iters_type(self, template):
        error_string = "invalid int value"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '  --learning_parameters.num_iters']
        cases = ["1.0", "-1", "Alpha"]
        for case in cases:
            temp = deepcopy(command_args)
            temp.append(case)
            ret = ote_train_common(template, root, temp)
            assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_num_iters_positive_case(self, template):
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.num_iters', '1']
        ret = ote_train_common(template, root, command_args)
        assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_num_iters_oob(self, template):
        error_string = "is out of bounds."
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '  --learning_parameters.num_iters']
        cases = ["0", "10001"]
        for case in cases:
            temp = deepcopy(command_args)
            temp.append(case)
            ret = ote_train_common(template, root, temp)
            assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_pp_confidence_threshold_type(self, template):
        error_string = "invalid float value"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--postprocessing.confidence_threshold']
        cases = ["-1", "Alpha"]
        for case in cases:
            temp = deepcopy(command_args)
            temp.append(case)
            ret = ote_train_common(template, root, temp)
            assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_confidence_threshold_positive_case(self, template):
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.confidence_threshold', '0,5']
        ret = ote_train_common(template, root, command_args)
        assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_pp_confidence_threshold_oob(self, template):
        error_string = "is out of bounds."
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--postprocessing.confidence_threshold',
                        '1.1'
                        ]
        ret = ote_train_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_pp_result_based_confidence_threshold(self, template):
        error_string = "Boolean value expected"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--postprocessing.result_based_confidence_threshold'
                        'NonBoolean']
        ret = ote_train_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_result_based_confidence_threshold_positive_case(self, template):
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.result_based_confidence_threshold',
                        'False']
        ret = ote_train_common(template, root, command_args)
        assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_nncf_opt_enable_quantization(self, template):
        error_string = "Boolean value expected"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--nncf_optimization.enable_quantization'
                        'NonBoolean']
        ret = ote_train_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_result_based_confidence_threshold_positive_case(self, template):
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.result_based_confidence_threshold',
                        'False']
        ret = ote_train_common(template, root, command_args)
        assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_enable_quantization_positive_case(self, template):
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.enable_quantization',
                        'False']
        ret = ote_train_common(template, root, command_args)
        assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_nncf_opt_enable_pruning(self, template):
        error_string = "Boolean value expected"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--nncf_optimization.enable_pruning'
                        'NonBoolean']
        ret = ote_train_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_lp_enable_pruning_positive_case(self, template):
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--learning_parameters.enable_pruning',
                        'False']
        ret = ote_train_common(template, root, command_args)
        assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_nncf_opt_maximal_accuracy_degradation_type(self, template):
        error_string = "invalid float value"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--nncf_optimization.maximal_accuracy_degradation']
        cases = ["-1", "Alpha"]
        for case in cases:
            temp = deepcopy(command_args)
            temp.append(case)
            ret = ote_train_common(template, root, temp)
            assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_nncf_opt_maximal_accuracy_degradation_oob(self, template):
        error_string = "is out of bounds"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--nncf_optimization.maximal_accuracy_degradation', '100.1']
        ret = ote_train_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_pp_maximal_confidence_threshold_type(self, template):
        error_string = "invalid float value"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--postprocessing.confidence_threshold']
        cases = ["-1", "Alpha"]
        for case in cases:
            temp = deepcopy(command_args)
            temp.append(case)
            ret = ote_train_common(template, root, temp)
            assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_pp_maximal_confidence_threshold_oob(self, template):
        error_string = "is out of bounds"
        command_args = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        'params',
                        '--postprocessing.confidence_threshold', '100.1']
        ret = ote_train_common(template, root, command_args)
        assert error_string in str(ret.stderr)
