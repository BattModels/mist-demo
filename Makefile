.ONESHELL:

.venv/: requirements.txt
	module load conda/2023-01-10-unstable
	conda activate base
	python3 -m venv --clear --system-site-packages $@
	source "$@/bin/activate"
	source submit/proxy_settings.sh
	python3 -m pip install --index-url https://download.pytorch.org/whl/cu118 torch
	python3 -m pip install --requirement $<
