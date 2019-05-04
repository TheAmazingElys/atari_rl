from __future__ import print_function

import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from contextlib import contextmanager

if __name__ == "__main__":

    # Use an existing experiment or create a new one
    mlflow.set_experiment("tracking_experiment")

    # A context manager to check if the run completed without error
    @contextmanager
    def check_completion(run):
        try:
            mlflow.log_param("State", "RUNNING")
            yield run
            mlflow.log_param("State", "COMPLETED")
        except:
            mlflow.log_param("State", "CRASHED")


    with check_completion(mlflow.start_run()) as active_run:

        X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0])
        lr = LogisticRegression()
        lr.fit(X, y)
        score = lr.score(X, y)

        # Log several metrics
        for i in range(10):
            mlflow.log_metric("i mod 2", i % 2)

        print("Score: %s" % score)
        mlflow.log_param("random", np.random.rand())
        mlflow.log_metric("score", score)
        mlflow.sklearn.log_model(lr, "model")
        print("Model saved in run %s" % active_run.info.run_uuid)
        mlflow.log_param("state", "COMPLETED")

