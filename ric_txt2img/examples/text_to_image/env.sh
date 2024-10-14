# export http_proxy="http://star-proxy.oa.com:3128"
# export https_proxy="http://star-proxy.oa.com:3128"
# export ftp_proxy="http://star-proxy.oa.com:3128"

# cd /mnt/aigc_cq/private/amandaaluo/own_tools/conda_set
# sh install_miniconda.sh
# export PATH="/root/miniconda3/bin:${PATH}"

# # conda create -y -n multi_obj PYTHON==3.10

# pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org

# # pip install torch==2.0.0 torchvision==0.15.1

pip install xformers==0.0.19

pip install -r requirements.txt

pip3 install deepspeed

cd /mnt/aigc_cq/private/amandaaluo/own_code/multi_objective/diffusers
pip install -e .