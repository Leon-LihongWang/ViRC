cd ./LLaMA-Factory
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -e ".[torch,metrics]"
pip install -r ../requirements.txt

cd ../verl
git checkout v0.5.0 --force
sed -i '29s/.*/        image["image"] = Image.open(BytesIO(image["bytes"]))/' ./verl/utils/dataset/vision_utils.py
pip install --no-deps -e .
