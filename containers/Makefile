.ONESHELL:

spack-mpich.def: spack.yaml templates/container/singularity.def
	spack -e . containerize > $@

spack-mpich.sif: spack-mpich.def training_container.sif
	spack load apptainer
	apptainer build --force $@ $<
