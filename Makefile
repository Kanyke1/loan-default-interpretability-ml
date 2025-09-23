PY=python -u

train_sklearn:
	$(PY) -m src.train_sklearn --config configs/sklearn_default.json

train_torch:
	$(PY) -m src.models.torch_mlp --config configs/torch_default.json

report:
	$(PY) -m src.eval.robustness --config configs/sklearn_default.json
	$(PY) -m src.eval.fairness --config configs/sklearn_default.json

test:
	pytest -q

format:
	black src tests
