test:
	python -Wd -m tests.test_discrete_random_variable

sdist:
	python setup.py sdist

docs:
	$(MAKE) html -C docs

.PHONY: test sdist docs
