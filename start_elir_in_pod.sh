#!/bin/bash


python3.12 -m venv .venv    # you can work with a different python version but this canvas was created using python3.12
source .venv/bin/activate
pip install --upgrade pip

cd ..
git clone https://github.com/eladc-git/ELIR.git
cd executorch

./install_executorch.sh
pip install diffusers


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

sed -i 's/^[[:space:]]*self\.to(x\.device)/# &/' "../ELIR/ELIR/models/elir.py"

cp schema/program.fbs exir/_serialize/
cp schema/scalar_type.fbs exir/_serialize/

python -m examples.arm.aot_arm_compiler \
  --model_name=elir \
  --delegate \
  --quantize \
  --output=simple_example.pte \
  --target=vgf \
  --intermediate=intermediate_artifacts_dir
  -s=pip-out/temp.linux-x86_64-cpython-312/cmake-out/kernels/quantized/libquantized_ops_aot_lib.so



sleep 12h
