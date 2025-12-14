#!/bin/bash


python3.12 -m venv .venv    # you can work with a different python version but this canvas was created using python3.12
source .venv/bin/activate
pip install --upgrade pip

cd ..
git clone https://github.com/eladc-git/ELIR.git
cd executorch

./install_executorch.sh
pin install diffusers


git config --global user.name "Mona Lisa"
git config --global user.email "monalisa@example.com"

examples/arm/setup.sh \
  --i-agree-to-the-contained-eula \
  --disable-ethos-u-deps \
  --enable-vgf-lib \
  --enable-model-converter

source examples/arm/ethos-u-scratch/setup_path.sh

which model-converter

export PYTHONPATH="$PYTHONPATH:$(pwd -P)/../ELIR"

python -m examples.arm.aot_arm_compiler \
  --model_name=elir \
  --delegate \
  --quantize \
  --output=simple_example.pte \
  --target=vgf \
  --intermediate=intermediate_artifacts_dir 



sleep 12h
