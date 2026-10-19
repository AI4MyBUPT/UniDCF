#!/usr/bin/env sh
pip install scikit-image==0.25.2 opencv-python==4.9.0.80 numpy==1.26.0
pip3 install torch_cluster-1.6.2+pt21cu118-cp310-cp310-linux_x86_64.whl
pip3 install torch_scatter-2.1.2+pt21cu118-cp310-cp310-linux_x86_64.whl

conda install pytorch3d-0.7.8-py310_cu118_pyt210.tar.bz2
 
python inference_UniDCF.py
python inference_UniDCF_mesh.py
python inference_baseline.py
python inference_baseline_mesh.py



