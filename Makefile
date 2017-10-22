test:
	python -Wd -m tests.test_discrete_random_variable

sdist:
	python setup.py sdist

upload:
	python setup.py sdist upload -r pypi

docs:
	$(MAKE) html -C docs

copy-docs:
	cp -r docs/_build/html/* ../pyitlib_pages

.PHONY: test sdist docs copy-docs
