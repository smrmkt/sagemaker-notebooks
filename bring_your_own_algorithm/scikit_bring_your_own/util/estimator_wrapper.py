# -*- coding: utf-8 -*-

class EstimatorWrapper():

    def __init__(self, estimator):
        self._estimator = estimator
    
    def fit(self, inputs):
        # ジョブ名は一意である必要がある
        from sagemaker.utils import base_name_from_image, name_from_base
        base_name = self._estimator.base_job_name or base_name_from_image(self._estimator.train_image())
        self._estimator._current_job_name = name_from_base(base_name)

        # アウトプットを出力する場所が指定されていない場合には，ここで指定
        if self._estimator.output_path is None:
            self._estimator.output_path = 's3://{}/'.format(self._estimator.sagemaker_session.default_bucket())

        from sagemaker.estimator import _TrainingJob
        self._estimator.latest_training_job = _TrainingJob.start_new(self._estimator, inputs)
    
    def set_hyperparameters(self, **kwargs):
        self._estimator.set_hyperparameters(**kwargs)

    def hyperparameters(self):
        return self._estimator.hyperparam_dict
